function summary = run_convergence(output_root, final_time, preset)
%RUN_CONVERGENCE 对指定双涡环工况做网格与时间步收敛性研究。
%
% 该函数会运行四组独立算例，时间较长。它把“空间加密”和“时间步减半”分开，
% 最终比较两次顺序交换的时间、末态能量和不可压残差。输出目录不可已有旧结果。

    if nargin < 3 || isempty(preset)
        preset = 'leapfrogging';
    end

    if nargin < 1 || isempty(output_root)
        matlab_root = fileparts(mfilename('fullpath'));
        run_stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        % 与主算例的中文目录名保持一致，避免结果目录只靠时间戳区分。
        switch lower(char(preset))
            case 'leapfrogging'
                experiment_name = '双涡环';
            case 'equal_rings'
                experiment_name = '等半径双涡环';
            case 'reference'
                experiment_name = '作者示例';
            otherwise
                error('未知 preset：%s。', preset);
        end
        output_root = fullfile(fileparts(matlab_root), 'outputs', ...
            [experiment_name, '_网格时间步收敛性_', run_stamp]);
    end
    if nargin < 2 || isempty(final_time)
        final_time = 85;
    end
    if ~isfolder(output_root)
        mkdir(output_root);
    end

    cases = struct( ...
        'name', {'space_coarse', 'baseline', 'space_fine', 'time_fine'}, ...
        'resolution', {[96, 48, 48], [128, 64, 64], [160, 80, 80], [128, 64, 64]}, ...
        'dt', {1 / 24, 1 / 24, 1 / 24, 1 / 48}, ...
        'time_factor', {1.0, 1.0, 1.0, 1.5});
    records = NaN(numel(cases), 9);
    for index = 1:numel(cases)
        config = make_config(preset, fullfile(output_root, cases(index).name), ...
            ceil(final_time * cases(index).time_factor / cases(index).dt));
        config.volume_resolution = cases(index).resolution;
        config.dt = cases(index).dt;
        config.snapshot_steps = unique(round(linspace(0, config.max_steps, 7)));
        config.tracking_stride = max(1, round((1 / 6) / config.dt));
        config.diagnostics_stride = config.tracking_stride;
        % 收敛性只比较可量化指标，不为每一组重复生成三维插图。
        config.visualize_at_end = false;
        result = run_simulation(config);
        final_diagnostics = result.diagnostics(end, :);
        records(index, :) = [index, result.cycle.first_exchange_time, ...
            result.cycle.order_return_time, final_diagnostics.total_energy, ...
            final_diagnostics.kinetic_energy, final_diagnostics.edge_divergence_l2, ...
            result.completed_steps * config.dt, config.max_steps * config.dt, ...
            double(result.cycle.order_cycle_detected)];
    end
    summary = array2table(records, 'VariableNames', {'case_index', ...
        'first_exchange_time', 'order_return_time', 'total_energy_at_stop', ...
        'kinetic_energy_at_stop', 'divergence_l2_at_stop', 'completed_time', ...
        'maximum_observation_time', 'order_cycle_detected'});
    summary.case_name = string({cases.name}).';
    summary = movevars(summary, 'case_name', 'Before', 'case_index');
    writetable(summary, fullfile(output_root, 'convergence_summary.csv'));
    plot_convergence(summary, output_root);
    fprintf('convergence study output: %s\n', output_root);
end

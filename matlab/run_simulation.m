function result = run_simulation(config)
%RUN_SIMULATION 运行 ISF 双涡环算例并保存完整数值证据。
%
% 计算步骤严格保持 Chern et al. Algorithm 1：SchroedingerFlow、Normalize、
% PressureProject。每隔 tracking_stride 步从 Algorithm 3 的离散相位涡量识别
% 两条环，并输出环心、半径、环量、能量、散度和交替穿越判据。

    validate_config(config);
    matlab_root = fileparts(mfilename('fullpath'));
    author_code_path = fullfile(matlab_root, 'reference');
    if ~isfolder(author_code_path)
        error('未找到 matlab/reference 中的作者基准代码。');
    end
    addpath(author_code_path);
    addpath(matlab_root);
    prepare_output_folder(config.output_path);

    isf = ISF(config.volume_size(1), config.volume_size(2), config.volume_size(3), ...
        config.volume_resolution(1), config.volume_resolution(2), config.volume_resolution(3));
    isf.hbar = config.hbar;
    isf.dt = config.dt;
    isf.BuildSchroedinger;
    [psi1, psi2] = initialise_two_rings(isf, config);
    material_markers = initialize_markers(config);

    expected_rings = [config.ring_centers(:, 1), config.ring_radii(:)];
    max_track_records = floor(config.max_steps / config.tracking_stride) + 2;
    max_diagnostic_records = floor(config.max_steps / config.diagnostics_stride) + 2;
    track_records = NaN(max_track_records, 18);
    material_records = NaN(max_track_records, 8);
    diagnostic_records = NaN(max_diagnostic_records, 6);
    track_count = 0;
    material_count = 0;
    diagnostic_count = 0;
    previous_rings = [];
    saved_event_count = 0;

    [track_count, previous_rings, track_records] = append_track( ...
        track_count, track_records, isf, psi1, psi2, 0, previous_rings, expected_rings);
    [material_count, material_records] = append_material_track( ...
        material_count, material_records, material_markers, config.volume_size, 0, 0.0);
    [diagnostic_count, diagnostic_records] = append_diagnostics( ...
        diagnostic_count, diagnostic_records, isf, psi1, psi2, 0, 0.0);
    save_state(config.output_path, 0, 0.0, psi1, psi2, config, material_markers);

    completed_steps = config.max_steps;
    for step = 1:config.max_steps
        [psi1, psi2] = isf.SchroedingerFlow(psi1, psi2);
        [psi1, psi2] = isf.Normalize(psi1, psi2);
        [psi1, psi2] = isf.PressureProject(psi1, psi2);
        material_markers = advect_markers(isf, psi1, psi2, material_markers);
        time = step * config.dt;

        is_tracking_step = mod(step, config.tracking_stride) == 0 || step == config.max_steps;
        cycle = struct('exchange_count', 0, 'order_cycle_detected', false);
        if is_tracking_step
            [track_count, previous_rings, track_records] = append_track( ...
                track_count, track_records, isf, psi1, psi2, step, previous_rings, expected_rings);
            [material_count, material_records] = append_material_track( ...
                material_count, material_records, material_markers, config.volume_size, step, time);
            current_material_tracking = material_tracking_table(material_records(1:material_count, :));
            cycle = detect_cycle(current_material_tracking, ...
                config.volume_size(1), isf.dx);
            if cycle.exchange_count > saved_event_count
                save_state(config.output_path, step, time, psi1, psi2, config, material_markers);
                saved_event_count = cycle.exchange_count;
            end
        end
        if mod(step, config.diagnostics_stride) == 0 || step == config.max_steps
            [diagnostic_count, diagnostic_records] = append_diagnostics( ...
                diagnostic_count, diagnostic_records, isf, psi1, psi2, step, time);
        end
        if any(config.snapshot_steps == step)
            save_state(config.output_path, step, time, psi1, psi2, config, material_markers);
        end
        if config.stop_after_order_return && is_tracking_step && ...
                cycle.order_cycle_detected && ...
                cycle.exchange_count >= config.minimum_exchange_count
            completed_steps = step;
            save_state(config.output_path, step, time, psi1, psi2, config, material_markers);
            break
        end
    end

    tracking = tracking_table(track_records(1:track_count, :));
    material_tracking = material_tracking_table(material_records(1:material_count, :));
    diagnostics = diagnostic_table(diagnostic_records(1:diagnostic_count, :));
    geometric_cycle = detect_cycle(tracking, config.volume_size(1), isf.dx);
    cycle = detect_cycle(material_tracking, config.volume_size(1), isf.dx);
    cycle.geometric_candidate_exchange_count = geometric_cycle.candidate_exchange_count;
    cycle.geometric_exchange_count = geometric_cycle.exchange_count;
    writetable(tracking, fullfile(config.output_path, 'ring_tracks.csv'));
    writetable(material_tracking, fullfile(config.output_path, 'material_ring_tracks.csv'));
    writetable(diagnostics, fullfile(config.output_path, 'solver_diagnostics.csv'));
    writetable(struct2table(cycle), fullfile(config.output_path, 'cycle_summary.csv'));
    save(fullfile(config.output_path, 'experiment_config.mat'), 'config', 'completed_steps', 'cycle');
    write_summary_text(config.output_path, config, completed_steps, cycle);
    plot_diagnostics(config.output_path, tracking, material_tracking, diagnostics, cycle);
    if config.visualize_at_end
        visualize_states(config.output_path);
    end

    result = struct('output_path', config.output_path, 'completed_steps', completed_steps, ...
        'tracking', tracking, 'material_tracking', material_tracking, ...
        'diagnostics', diagnostics, 'cycle', cycle);
    fprintf('ISF experiment output: %s\n', config.output_path);
end

function validate_config(config)
    required = {'output_path', 'max_steps', 'volume_size', 'volume_resolution', ...
        'hbar', 'dt', 'background_velocity', 'ring_centers', 'ring_radii', ...
        'ring_normal', 'snapshot_steps', 'tracking_stride', 'diagnostics_stride', ...
        'minimum_exchange_count', 'material_marker_count', 'stop_after_order_return', ...
        'visualize_at_end'};
    for index = 1:numel(required)
        if ~isfield(config, required{index})
            error('实验参数缺少字段：%s。', required{index});
        end
    end
    if config.max_steps < 1 || config.max_steps ~= floor(config.max_steps)
        error('max_steps 必须为正整数。');
    end
    if size(config.ring_centers, 1) ~= 2 || numel(config.ring_radii) ~= 2
        error('当前任务只支持两条涡环，ring_centers 和 ring_radii 必须各含两项。');
    end
end

function prepare_output_folder(output_path)
    if ~isfolder(output_path)
        mkdir(output_path);
        return
    end
    old_artifacts = [dir(fullfile(output_path, 'state_*.mat')); ...
        dir(fullfile(output_path, 'ring_tracks.csv')); ...
        dir(fullfile(output_path, 'matlab_solver_diagnostics.csv'))];
    if ~isempty(old_artifacts)
        error(['输出目录已有旧计算结果。为避免新旧数据混合，请指定一个新的输出目录；', ...
            '本程序不会自动删除任何结果。']);
    end
end

function [psi1, psi2] = initialise_two_rings(isf, config)
    phase = config.background_velocity(1) / config.hbar * isf.px + ...
        config.background_velocity(2) / config.hbar * isf.py + ...
        config.background_velocity(3) / config.hbar * isf.pz;
    psi1 = exp(1i * phase);
    psi2 = 0.01 * exp(1i * phase);
    disk_thickness = 5 * isf.dx;
    for ring_index = 1:2
        psi1 = isf.AddCircle(psi1, config.ring_centers(ring_index, :), ...
            config.ring_normal, config.ring_radii(ring_index), disk_thickness);
    end
    [psi1, psi2] = isf.Normalize(psi1, psi2);
    [psi1, psi2] = isf.PressureProject(psi1, psi2);
end

function [count, rings, records] = append_track(count, records, isf, psi1, psi2, step, previous_rings, expected_rings)
    [rings, ~] = track_rings(isf, psi1, psi2, previous_rings, expected_rings);
    if rings.available
        if isempty(previous_rings) || ~isfield(previous_rings, 'available') || ...
                ~previous_rings.available || ~isfield(previous_rings, 'x1_unwrapped')
            rings.x1_unwrapped = rings.x1;
            rings.x2_unwrapped = rings.x2;
        else
            rings.x1_unwrapped = previous_rings.x1_unwrapped + ...
                periodic_increment(rings.x1 - previous_rings.x1, isf.sizex);
            rings.x2_unwrapped = previous_rings.x2_unwrapped + ...
                periodic_increment(rings.x2 - previous_rings.x2, isf.sizex);
        end
    else
        rings.x1_unwrapped = NaN;
        rings.x2_unwrapped = NaN;
    end
    count = count + 1;
    records(count, :) = [step, step * isf.dt, rings.x1, rings.x1_unwrapped, ...
        rings.radius1, rings.x2, rings.x2_unwrapped, rings.radius2, ...
        rings.circulation1, rings.circulation2, rings.raw_winding1, ...
        rings.raw_winding2, rings.integer_winding1, rings.integer_winding2, ...
        rings.peak_distance, rings.peak_strength_ratio, rings.matching_cost, ...
        double(rings.resolved)];
    % 合并帧只用于记录，不用作下一帧的身份匹配参考；否则一次峰值跳变会把
    % 错误编号永久传递到后续轨迹。
    if ~rings.resolved && ~isempty(previous_rings) && ...
            isfield(previous_rings, 'available') && previous_rings.available
        rings = previous_rings;
    end
end

function increment = periodic_increment(delta, period)
    increment = mod(delta + 0.5 * period, period) - 0.5 * period;
end

function [count, records] = append_diagnostics(count, records, isf, psi1, psi2, step, time)
    values = compute_diagnostics(isf, psi1, psi2);
    count = count + 1;
    records(count, :) = [step, time, values.edge_divergence_l2, values.total_energy, ...
        values.kinetic_energy, values.density_constraint_l2];
end

function [count, records] = append_material_track(count, records, markers, volume_size, step, time)
    values = measure_markers(markers, volume_size);
    count = count + 1;
    records(count, :) = [step, time, values.x1, values.radius1, values.radius_std1, ...
        values.x2, values.radius2, values.radius_std2];
end

function table_data = tracking_table(records)
    table_data = array2table(records, 'VariableNames', {'step', 'time', ...
        'ring_1_x', 'ring_1_x_unwrapped', 'ring_1_radius', ...
        'ring_2_x', 'ring_2_x_unwrapped', 'ring_2_radius', ...
        'ring_1_circulation', 'ring_2_circulation', ...
        'ring_1_raw_winding', 'ring_2_raw_winding', ...
        'ring_1_integer_winding', 'ring_2_integer_winding', ...
        'peak_distance', 'peak_strength_ratio', 'matching_cost', 'track_resolved'});
end

function table_data = diagnostic_table(records)
    table_data = array2table(records, 'VariableNames', {'step', 'time', ...
        'edge_divergence_l2', 'total_energy', 'kinetic_energy', 'density_constraint_l2'});
end

function table_data = material_tracking_table(records)
    table_data = array2table(records, 'VariableNames', {'step', 'time', ...
        'ring_1_x_unwrapped', 'ring_1_material_radius', 'ring_1_radius_std', ...
        'ring_2_x_unwrapped', 'ring_2_material_radius', 'ring_2_radius_std'});
    table_data.track_resolved = double( ...
        table_data.ring_1_radius_std <= 0.40 * table_data.ring_1_material_radius & ...
        table_data.ring_2_radius_std <= 0.40 * table_data.ring_2_material_radius);
    % 为 detect_cycle 的统一接口提供未展开坐标别名；材料 x 本身已连续。
    table_data.ring_1_x = table_data.ring_1_x_unwrapped;
    table_data.ring_2_x = table_data.ring_2_x_unwrapped;
end

function save_state(output_path, step, time, psi1, psi2, config, material_markers)
% 保存关键状态供等值面后处理；配置一并写入，保证图片可追溯到数值参数。
    filename = fullfile(output_path, sprintf('state_%05d.mat', step));
    volume_size = config.volume_size;
    volume_resolution = config.volume_resolution;
    hbar = config.hbar;
    dt = config.dt;
    ring_centers = config.ring_centers;
    ring_radii = config.ring_radii;
    background_velocity = config.background_velocity;
    material_marker_positions = cell(1, numel(material_markers));
    for ring_index = 1:numel(material_markers)
        material_marker_positions{ring_index} = struct('x', material_markers{ring_index}.x, ...
            'y', material_markers{ring_index}.y, 'z', material_markers{ring_index}.z);
    end
    save(filename, 'step', 'time', 'psi1', 'psi2', 'volume_size', ...
        'volume_resolution', 'hbar', 'dt', 'ring_centers', 'ring_radii', ...
        'background_velocity', 'material_marker_positions', '-v7.3');
end

function write_summary_text(output_path, config, completed_steps, cycle)
    filename = fullfile(output_path, 'leapfrogging_summary.txt');
    file_identifier = fopen(filename, 'w');
    if file_identifier < 0
        error('无法写入 %s。', filename);
    end
    cleaner = onCleanup(@() fclose(file_identifier));
    fprintf(file_identifier, 'label: %s\n', config.label);
    fprintf(file_identifier, 'completed_steps: %d\n', completed_steps);
    fprintf(file_identifier, 'completed_time: %.12g\n', completed_steps * config.dt);
    fprintf(file_identifier, 'candidate_exchange_count: %d\n', cycle.candidate_exchange_count);
    fprintf(file_identifier, 'exchange_count: %d\n', cycle.exchange_count);
    fprintf(file_identifier, 'first_exchange_time: %.12g\n', cycle.first_exchange_time);
    fprintf(file_identifier, 'second_exchange_time: %.12g\n', cycle.second_exchange_time);
    fprintf(file_identifier, 'order_cycle_detected: %d\n', cycle.order_cycle_detected);
    fprintf(file_identifier, 'unresolved_fraction: %.12g\n', cycle.unresolved_fraction);
    fprintf(file_identifier, 'axial_overlap_fraction: %.12g\n', cycle.axial_overlap_fraction);
    fprintf(file_identifier, 'geometric_candidate_exchange_count: %d\n', cycle.geometric_candidate_exchange_count);
    fprintf(file_identifier, 'geometric_exchange_count: %d\n', cycle.geometric_exchange_count);
end

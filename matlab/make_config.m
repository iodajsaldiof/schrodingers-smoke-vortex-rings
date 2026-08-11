function config = make_config(preset, output_path, max_steps)
%MAKE_CONFIG 建立可复现的双涡环实验参数。
%
% 该函数把“题目工况”和“作者示例工况”明确分开：前者用于最终作业，后者
% 用于核对 Chern et al. 附带 MATLAB 示例。两者都使用同一 ISF 求解器和诊断。

    if nargin < 1 || isempty(preset)
        preset = 'equal_rings';
    end
    if nargin < 2
        output_path = '';
    end
    if nargin < 3 || isempty(max_steps)
        max_steps = 2000;
    end

    matlab_root = fileparts(mfilename('fullpath'));
    repository_root = fileparts(matlab_root);
    preset = lower(char(preset));
    switch preset
        case 'reference'
            config.label = 'Chern et al. author reference: example_leapfrog';
            config.volume_size = [10, 5, 5];
            config.volume_resolution = [128, 64, 64];
            config.hbar = 0.1;
            config.dt = 1 / 24;
            config.background_velocity = [-0.2, 0, 0];
            % 作者 example_leapfrog.m：两条环共心、半径不同。
            config.ring_centers = [5, 2.5, 2.5; 5, 2.5, 2.5];
            config.ring_radii = [1.5, 0.9];
            config.ring_normal = [-1, 0, 0];
            config.stop_after_order_return = false;
            % 默认输出目录使用中文任务名，便于在 outputs 中快速定位。
            default_name = '作者示例';
        case 'leapfrogging'
            % Chern et al. 的 example_leapfrog.m / Figure 4 基准参数。
            % 该工况是本项目展示完整交替穿越周期的主算例。
            config.label = 'Chern et al. Figure 4: leapfrogging vortex rings';
            config.volume_size = [10, 5, 5];
            config.volume_resolution = [128, 64, 64];
            config.hbar = 0.1;
            config.dt = 1 / 24;
            config.background_velocity = [-0.2, 0, 0];
            config.ring_centers = [5, 2.5, 2.5; 5, 2.5, 2.5];
            config.ring_radii = [1.5, 0.9];
            config.ring_normal = [-1, 0, 0];
            config.stop_after_order_return = true;
            default_name = '双涡环完整交替穿越';
        case 'equal_rings'
            config.label = 'B task 2: coaxial equal-radius vortex rings';
            config.volume_size = [10, 5, 5];
            config.volume_resolution = [128, 64, 64];
            config.hbar = 0.1;
            config.dt = 1 / 24;
            config.background_velocity = [0, 0, 0];
            % 科研实践 B 题建议参数：R=1，轴向间距 d=0.6。
            config.ring_centers = [4.7, 2.5, 2.5; 5.3, 2.5, 2.5];
            config.ring_radii = [1.0, 1.0];
            config.ring_normal = [-1, 0, 0];
            config.stop_after_order_return = true;
            default_name = '等半径双涡环';
        otherwise
            error(['未知 preset：%s。可用值为 reference、leapfrogging ', ...
                '或 equal_rings。'], preset);
    end

    if isempty(output_path)
        run_stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        run_name = sprintf('%s_%s', default_name, run_stamp);
        output_path = fullfile(repository_root, 'outputs', run_name);
    end
    config.output_path = char(output_path);
    config.preset = preset;
    config.max_steps = max_steps;
    config.snapshot_steps = unique([0, 45, 360, 720, 1080, 1440, 1800, max_steps]);
    config.tracking_stride = 4;
    if strcmp(preset, 'leapfrogging')
        % Figure 4 中环的相互穿越较快，使用更密的轨迹采样减少判据插值误差。
        config.tracking_stride = 2;
    end
    config.diagnostics_stride = 4;
    config.minimum_exchange_count = 2;
    config.material_marker_count = 96;
    config.visualize_at_end = true;
end

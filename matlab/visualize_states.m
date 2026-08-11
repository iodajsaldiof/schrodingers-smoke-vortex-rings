function visualize_states(output_path, phase_vorticity_level, selected_steps)
%VISUALIZE_STATES 将 ISF 关键帧渲染为可比较的涡环图像。
%
% 输入 output_path 是 run_reference 保存 state_*.mat 的目录。函数使用
% Algorithm 3 的边相位环量计算离散涡量，而不是直接画 |psi2|：作者的相位圆盘
% 初值中 |psi2| 仅是 0.01 的防零保护，拓扑涡核主要编码在波函数相位中。
%
% 输出：
%   visuals/state_XXXXX_phase_vorticity.png  每个关键时刻的三维涡核等值面
%   visuals/phase_vorticity_profiles_absolute.png  共享绝对色标的强度对比图
%   visuals/phase_vorticity_profiles_normalized.png 每帧归一化的几何形态图
%   visuals/vortex_visualization_summary.csv 固定等值面阈值与每帧峰值
%
% 可选 selected_steps 可只渲染指定步数，例如 [0, 360]，便于先快速检查。

    if nargin < 1 || isempty(output_path)
        % run_equal_rings 会创建带时间戳的新目录；这里必须递归搜索，不能只检查
        % 旧版 matlab_reference 这个固定目录。
        matlab_root = fileparts(mfilename('fullpath'));
        repository_root = fileparts(matlab_root);
        output_path = find_latest_state_output(repository_root, matlab_root);
        if isempty(output_path)
            error(['未在仓库 outputs 目录下找到 state_*.mat。请先运行 run_equal_rings，', ...
                '或把 result.output_path 作为第一个参数传入。']);
        end
    end
    state_files = dir(fullfile(output_path, 'state_*.mat'));
    if isempty(state_files)
        error('输出目录中没有可视化所需的 state_*.mat 文件。');
    end
    [~, order] = sort({state_files.name});
    state_files = state_files(order);
    if nargin >= 3 && ~isempty(selected_steps)
        available_steps = zeros(numel(state_files), 1);
        for index = 1:numel(state_files)
            tokens = sscanf(state_files(index).name, 'state_%d.mat');
            available_steps(index) = tokens(1);
        end
        state_files = state_files(ismember(available_steps, selected_steps));
        if isempty(state_files)
            error('selected_steps 中没有已保存的状态文件。');
        end
    end

    visuals_path = fullfile(output_path, 'visuals');
    if ~isfolder(visuals_path)
        mkdir(visuals_path);
    end

    initial = load(fullfile(state_files(1).folder, state_files(1).name));
    [initial_core, grid] = phase_vorticity_magnitude(initial);
    if nargin < 2 || isempty(phase_vorticity_level)
        % 采用所有已选关键帧峰值中的最小值的 10%。这样阈值对所有帧一致，
        % 同时不会因初始瞬时峰值过高而把后续涡核全部裁掉。
        peak_values = zeros(numel(state_files), 1);
        peak_values(1) = max(initial_core(:));
        for index = 2:numel(state_files)
            data = load(fullfile(state_files(index).folder, state_files(index).name));
            core = phase_vorticity_magnitude(data);
            peak_values(index) = max(core(:));
        end
        phase_vorticity_level = 0.10 * min(peak_values);
    end
    if phase_vorticity_level <= 0.0
        error('离散相位涡量等值面阈值必须为正数。');
    end

    count = numel(state_files);
    profiles = cell(count, 1);
    normalized_profiles = cell(count, 1);
    profile_maximum = 0.0;
    records = zeros(count, 5);
    for index = 1:count
        data = load(fullfile(state_files(index).folder, state_files(index).name));
        [core, current_grid] = phase_vorticity_magnitude(data);
        [profile, radial_coordinate] = axial_radial_profile(core, current_grid);
        profiles{index} = profile;
        profile_maximum = max(profile_maximum, max(profile(:)));
        normalized_profiles{index} = profile / max(max(profile(:)), eps);

        has_surface = max(core(:)) > phase_vorticity_level;
        records(index, :) = [data.step, data.time, max(core(:)), ...
            phase_vorticity_level, double(has_surface)];
        if has_surface
            material_marker_positions = [];
            if isfield(data, 'material_marker_positions')
                material_marker_positions = data.material_marker_positions;
            end
            save_isosurface(core, current_grid, phase_vorticity_level, ...
                fullfile(visuals_path, sprintf('state_%05d_phase_vorticity.png', data.step)), ...
                sprintf('Discrete phase-vorticity core, step %d, t = %.3f', ...
                data.step, data.time), material_marker_positions);
        else
            warning('第 %d 步的峰值低于固定等值面阈值，未生成三维表面。', data.step);
        end
    end

    save_profiles(profiles, state_files, grid, radial_coordinate, profile_maximum, ...
        'Absolute discrete phase-vorticity profiles', ...
        fullfile(visuals_path, 'phase_vorticity_profiles_absolute.png'));
    save_profiles(normalized_profiles, state_files, grid, radial_coordinate, 1.0, ...
        'Per-frame normalized phase-vorticity profiles', ...
        fullfile(visuals_path, 'phase_vorticity_profiles_normalized.png'));
    summary = array2table(records, 'VariableNames', ...
        {'step', 'time', 'phase_vorticity_max', 'fixed_isosurface_level', ...
        'isosurface_written'});
    writetable(summary, fullfile(visuals_path, 'vortex_visualization_summary.csv'));
    fprintf('vortex visualizations: %s\n', visuals_path);
end

function [magnitude, grid] = phase_vorticity_magnitude(data)
% 由作者 Algorithm 3 所用的边相位一形式离散外微分得到涡核标量场。
    psi1 = data.psi1;
    psi2 = data.psi2;
    lengths = data.volume_size;
    resolution = data.volume_resolution;
    dx = lengths(1) / resolution(1);
    dy = lengths(2) / resolution(2);
    dz = lengths(3) / resolution(3);

    phase_x = angle(conj(psi1) .* circshift(psi1, [-1, 0, 0]) + ...
        conj(psi2) .* circshift(psi2, [-1, 0, 0]));
    phase_y = angle(conj(psi1) .* circshift(psi1, [0, -1, 0]) + ...
        conj(psi2) .* circshift(psi2, [0, -1, 0]));
    phase_z = angle(conj(psi1) .* circshift(psi1, [0, 0, -1]) + ...
        conj(psi2) .* circshift(psi2, [0, 0, -1]));

    curl_x = phase_y + circshift(phase_z, [0, -1, 0]) - ...
        circshift(phase_y, [0, 0, -1]) - phase_z;
    curl_y = phase_z + circshift(phase_x, [0, 0, -1]) - ...
        circshift(phase_z, [-1, 0, 0]) - phase_x;
    curl_z = phase_x + circshift(phase_y, [-1, 0, 0]) - ...
        circshift(phase_x, [0, -1, 0]) - phase_y;
    omega_x = data.hbar * curl_x / (dy * dz);
    omega_y = data.hbar * curl_y / (dz * dx);
    omega_z = data.hbar * curl_z / (dx * dy);
    magnitude = sqrt(omega_x.^2 + omega_y.^2 + omega_z.^2);

    x = (0:resolution(1)-1) * dx;
    y = (0:resolution(2)-1) * dy;
    z = (0:resolution(3)-1) * dz;
    [grid.x, grid.y, grid.z] = ndgrid(x, y, z);
    grid.lengths = lengths;
    grid.resolution = resolution;
end

function save_isosurface(core, grid, level, filename, title_text, material_marker_positions)
% 统一相机与坐标范围；若状态包含材料标记，则叠加两种颜色展示涡环身份。
    figure_handle = figure('Visible', 'off', 'Color', 'w', ...
        'Position', [80, 80, 920, 650]);
    surface = patch(isosurface(grid.x, grid.y, grid.z, core, level));
    set(surface, 'FaceColor', [0.85, 0.33, 0.10], 'EdgeColor', 'none', ...
        'FaceAlpha', 0.82);
    hold on;
    if ~isempty(material_marker_positions)
        marker_colors = [0.00, 0.75, 1.00; 0.82, 0.10, 0.95];
        marker_handles = gobjects(1, min(numel(material_marker_positions), 2));
        for ring_index = 1:min(numel(material_marker_positions), 2)
            marker = material_marker_positions{ring_index};
            marker_handles(ring_index) = scatter3(mod(marker.x, grid.lengths(1)), ...
                mod(marker.y, grid.lengths(2)), mod(marker.z, grid.lengths(3)), 28, ...
                marker_colors(ring_index, :), 'filled', 'MarkerEdgeColor', 'w', ...
                'LineWidth', 0.5);
        end
        legend(marker_handles, {'ring 1 material markers', 'ring 2 material markers'}, ...
            'Location', 'northeast');
    end
    hold off;
    axis([0, grid.lengths(1), 0, grid.lengths(2), 0, grid.lengths(3)]);
    daspect([1, 1, 1]);
    view(-56, 23);
    camlight('headlight');
    lighting gouraud;
    set(gca, 'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');
    xlabel('x'); ylabel('y'); zlabel('z');
    title(title_text, 'Interpreter', 'none');
    print(figure_handle, filename, '-dpng', '-r220');
    close(figure_handle);
end

function [profile, radial_coordinate] = axial_radial_profile(core, grid)
% 把三维涡核压缩到 (x,r)，清楚呈现同轴涡环在轴向和半径方向上的演化。
    radial_limit = 0.5 * min(grid.lengths(2), grid.lengths(3));
    bin_count = max(16, min(64, floor(min(grid.resolution(2:3)) / 2)));
    radial_width = radial_limit / bin_count;
    radius = sqrt((grid.y - 0.5 * grid.lengths(2)).^2 + ...
        (grid.z - 0.5 * grid.lengths(3)).^2);
    radial_index = min(floor(radius / radial_width) + 1, bin_count);
    profile = zeros(grid.resolution(1), bin_count);
    for x_index = 1:grid.resolution(1)
        values = reshape(core(x_index, :, :), [], 1);
        indices = reshape(radial_index(x_index, :, :), [], 1);
        profile(x_index, :) = accumarray(indices, values, [bin_count, 1], ...
            @mean, 0).';
    end
    radial_coordinate = ((1:bin_count) - 0.5) * radial_width;
end

function save_profiles(profiles, state_files, grid, radial_coordinate, profile_maximum, title_text, filename)
% 一个面板对应一个关键帧；调用方选择绝对或归一化色标，分别回答强度和几何问题。
    count = numel(profiles);
    column_count = min(2, count);
    row_count = ceil(count / column_count);
    figure_handle = figure('Visible', 'off', 'Color', 'w', ...
        'Position', [80, 80, 980, 340 * row_count]);
    layout = tiledlayout(row_count, column_count, 'TileSpacing', 'compact', ...
        'Padding', 'compact');
    for index = 1:count
        data = load(fullfile(state_files(index).folder, state_files(index).name), ...
            'step', 'time');
        axis_handle = nexttile(layout);
        imagesc(radial_coordinate, (0:grid.resolution(1)-1) * ...
            grid.lengths(1) / grid.resolution(1), profiles{index});
        set(axis_handle, 'YDir', 'normal');
        clim(axis_handle, [0, profile_maximum]);
        xlabel('radius r'); ylabel('axial coordinate x');
        title(sprintf('step %d, t = %.3f', data.step, data.time));
        colorbar;
    end
    title(layout, title_text);
    print(figure_handle, filename, '-dpng', '-r220');
    close(figure_handle);
end

function output_path = find_latest_state_output(repository_root, matlab_root)
% 优先选择完整交替穿越主算例；若不存在，再回退到最新的状态目录。
    search_roots = {fullfile(repository_root, 'outputs'), ...
        fullfile(matlab_root, 'outputs')};
    latest_time = -Inf;
    latest_primary_time = -Inf;
    output_path = '';
    primary_output_path = '';
    for root_index = 1:numel(search_roots)
        search_root = search_roots{root_index};
        if ~isfolder(search_root)
            continue
        end
        matches = dir(fullfile(search_root, '**', 'state_*.mat'));
        for match_index = 1:numel(matches)
            candidate_time = matches(match_index).datenum;
            if candidate_time > latest_time
                latest_time = candidate_time;
                output_path = matches(match_index).folder;
            end
            candidate_path = matches(match_index).folder;
            is_primary = contains(candidate_path, '完整交替穿越') || ...
                (contains(candidate_path, 'leapfrogging') && ...
                ~contains(candidate_path, 'convergence'));
            if is_primary && candidate_time > latest_primary_time
                latest_primary_time = candidate_time;
                primary_output_path = candidate_path;
            end
        end
    end
    if ~isempty(primary_output_path)
        output_path = primary_output_path;
    end
end

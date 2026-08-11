function [rings, profile] = track_rings(isf, psi1, psi2, previous_rings, initial_rings)
%TRACK_RINGS 在 (x,r) 图中提取两条同轴环，并显式给出可信度。
%
% 仅凭涡量模长的两个局部峰不能保证它们仍是两条独立的涡环。根据论文 6 的
% 提醒，本函数把“有两个数值峰”与“可分辨的两条几何涡环”区分开：当峰的
% (x,r) 距离不足一个可分辨尺度时，resolved=false，后续周期判据不会使用该帧。
% 该轻量方案面向本题的轴对称情形；复杂三维流才需要论文 6/7 的全局涡丝骨架。

    if nargin < 4
        previous_rings = [];
    end
    if nargin < 5 || isempty(initial_rings)
        initial_rings = [isf.sizex / 2, 0.9; isf.sizex / 2, 1.5];
    end

    [profile, x_coordinate, radial_coordinate] = phase_vorticity_profile(isf, psi1, psi2);
    candidates = extract_peak_candidates(profile, x_coordinate, radial_coordinate, isf.sizex);
    rings = empty_ring_record();
    if size(candidates, 1) < 2
        return
    end

    if isempty(previous_rings) || ~isfield(previous_rings, 'available') || ...
            ~previous_rings.available
        targets = initial_rings;
    else
        targets = [previous_rings.x1, previous_rings.radius1; ...
                   previous_rings.x2, previous_rings.radius2];
    end
    [first_index, second_index, matching_cost] = match_two_rings( ...
        candidates, targets, isf.sizex, isf.dx, radial_coordinate);
    if isnan(first_index)
        return
    end

    rings.x1 = candidates(first_index, 1);
    rings.radius1 = candidates(first_index, 2);
    rings.peak1 = candidates(first_index, 3);
    rings.x2 = candidates(second_index, 1);
    rings.radius2 = candidates(second_index, 2);
    rings.peak2 = candidates(second_index, 3);
    rings.matching_cost = matching_cost;
    rings.available = true;

    radial_spacing = radial_coordinate(2) - radial_coordinate(1);
    x_separation = periodic_distance(rings.x1, rings.x2, isf.sizex);
    r_separation = abs(rings.radius1 - rings.radius2);
    rings.peak_distance = sqrt((x_separation / (4 * isf.dx))^2 + ...
        (r_separation / (4 * radial_spacing))^2);
    rings.peak_strength_ratio = min(rings.peak1, rings.peak2) / ...
        max(rings.peak1, rings.peak2);
    % 小于 1.25 表示两峰在网格上没有足够的分离，身份匹配不具物理意义。
    rings.resolved = rings.peak_distance >= 1.25 && ...
        rings.peak_strength_ratio >= 0.08;

    [rings.raw_winding1, rings.integer_winding1, rings.circulation1] = ...
        square_loop_winding(isf, psi1, psi2, rings.x1, rings.radius1, rings.resolved);
    [rings.raw_winding2, rings.integer_winding2, rings.circulation2] = ...
        square_loop_winding(isf, psi1, psi2, rings.x2, rings.radius2, rings.resolved);
end

function rings = empty_ring_record()
    rings = struct('x1', NaN, 'radius1', NaN, 'peak1', NaN, ...
        'x2', NaN, 'radius2', NaN, 'peak2', NaN, ...
        'circulation1', NaN, 'circulation2', NaN, ...
        'raw_winding1', NaN, 'raw_winding2', NaN, ...
        'integer_winding1', NaN, 'integer_winding2', NaN, ...
        'peak_distance', NaN, 'peak_strength_ratio', NaN, ...
        'matching_cost', NaN, 'available', false, 'resolved', false);
end

function [profile, x_coordinate, radial_coordinate] = phase_vorticity_profile(isf, psi1, psi2)
% 与 Algorithm 3 相同的边相位一形式；其离散外微分给出相位涡量指标。
    phase_x = angle(conj(psi1) .* circshift(psi1, [-1, 0, 0]) + ...
        conj(psi2) .* circshift(psi2, [-1, 0, 0]));
    phase_y = angle(conj(psi1) .* circshift(psi1, [0, -1, 0]) + ...
        conj(psi2) .* circshift(psi2, [0, -1, 0]));
    phase_z = angle(conj(psi1) .* circshift(psi1, [0, 0, -1]) + ...
        conj(psi2) .* circshift(psi2, [0, 0, -1]));
    [omega_x, omega_y, omega_z] = isf.DerivativeOfOneForm(phase_x, phase_y, phase_z);
    omega_x = isf.hbar * omega_x / (isf.dy * isf.dz);
    omega_y = isf.hbar * omega_y / (isf.dz * isf.dx);
    omega_z = isf.hbar * omega_z / (isf.dx * isf.dy);
    magnitude = sqrt(omega_x.^2 + omega_y.^2 + omega_z.^2);

    bin_count = max(20, min(64, floor(min(isf.resy, isf.resz) / 2)));
    radial_limit = 0.5 * min(isf.sizey, isf.sizez);
    radial_width = radial_limit / bin_count;
    radius = sqrt((isf.py - 0.5 * isf.sizey).^2 + ...
        (isf.pz - 0.5 * isf.sizez).^2);
    radial_index = min(floor(radius / radial_width) + 1, bin_count);
    profile = zeros(isf.resx, bin_count);
    for x_index = 1:isf.resx
        values = reshape(magnitude(x_index, :, :), [], 1);
        indices = reshape(radial_index(x_index, :, :), [], 1);
        profile(x_index, :) = accumarray(indices, values, [bin_count, 1], @mean, 0).';
    end
    x_coordinate = (0:isf.resx - 1) * isf.dx;
    radial_coordinate = ((1:bin_count) - 0.5) * radial_width;
end

function candidates = extract_peak_candidates(profile, x_coordinate, radial_coordinate, x_period)
% 非极大值抑制避免把同一涡环的相邻格点错误计为两条环。
    kernel = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16;
    smoothed = conv2(profile, kernel, 'same');
    [values, order] = sort(smoothed(:), 'descend');
    candidates = zeros(0, 3);
    if isempty(values) || values(1) <= 0
        return
    end
    x_spacing = max(0.18, 2 * (x_coordinate(2) - x_coordinate(1)));
    r_spacing = max(0.14, 2 * (radial_coordinate(2) - radial_coordinate(1)));
    for entry = 1:numel(order)
        if values(entry) < 0.03 * values(1) || size(candidates, 1) >= 12
            break
        end
        [x_index, r_index] = ind2sub(size(smoothed), order(entry));
        candidate = [x_coordinate(x_index), radial_coordinate(r_index), values(entry)];
        accept = true;
        for chosen = 1:size(candidates, 1)
            dx = periodic_distance(candidate(1), candidates(chosen, 1), x_period);
            dr = candidate(2) - candidates(chosen, 2);
            if (dx / x_spacing)^2 + (dr / r_spacing)^2 < 1.0
                accept = false;
                break
            end
        end
        if accept
            candidates(end + 1, :) = candidate; %#ok<AGROW>
        end
    end
end

function [first_index, second_index, best_cost] = match_two_rings(candidates, targets, x_period, dx, radial_coordinate)
% 穷举两候选峰和两种编号，以前一可信帧为参考维持身份连续。
    first_index = NaN;
    second_index = NaN;
    best_cost = Inf;
    x_scale = max(0.35, 4 * dx);
    r_scale = max(0.20, 4 * (radial_coordinate(2) - radial_coordinate(1)));
    for first = 1:size(candidates, 1) - 1
        for second = first + 1:size(candidates, 1)
            pair = [first, second; second, first];
            for permutation = 1:2
                current = candidates(pair(permutation, :), 1:2);
                x_cost = periodic_distance(current(:, 1), targets(:, 1), x_period) / x_scale;
                r_cost = (current(:, 2) - targets(:, 2)) / r_scale;
                cost = sum(x_cost.^2 + r_cost.^2);
                if cost < best_cost
                    best_cost = cost;
                    first_index = pair(permutation, 1);
                    second_index = pair(permutation, 2);
                end
            end
        end
    end
end

function distance = periodic_distance(a, b, period)
    distance = abs(mod((a - b) + 0.5 * period, period) - 0.5 * period);
end

function [raw_winding, integer_winding, circulation] = square_loop_winding(isf, psi1, psi2, center_x, radius, is_resolved)
% 计算围绕涡核的相位绕数；只在两环可分辨时报告量子化环量 2*pi*hbar*n。
    phase_x = angle(conj(psi1) .* circshift(psi1, [-1, 0, 0]) + ...
        conj(psi2) .* circshift(psi2, [-1, 0, 0]));
    phase_y = angle(conj(psi1) .* circshift(psi1, [0, -1, 0]) + ...
        conj(psi2) .* circshift(psi2, [0, -1, 0]));
    ix_center = coordinate_to_index(center_x, isf.dx, isf.resx);
    iy_center = coordinate_to_index(0.5 * isf.sizey + radius, isf.dy, isf.resy);
    iz_center = coordinate_to_index(0.5 * isf.sizez, isf.dz, isf.resz);
    half_width = max(2, round(0.25 / min(isf.dx, isf.dy)));
    x_indices = wrap_indices(ix_center - half_width, 2 * half_width + 1, isf.resx);
    y_indices = wrap_indices(iy_center - half_width, 2 * half_width + 1, isf.resy);
    phase_sum = 0.0;
    for index = 1:2 * half_width
        phase_sum = phase_sum + phase_x(x_indices(index), y_indices(1), iz_center);
        phase_sum = phase_sum + phase_y(x_indices(end), y_indices(index), iz_center);
        phase_sum = phase_sum - phase_x(x_indices(index), y_indices(end), iz_center);
        phase_sum = phase_sum - phase_y(x_indices(1), y_indices(index), iz_center);
    end
    raw_winding = phase_sum / (2 * pi);
    integer_winding = round(raw_winding);
    winding_error = abs(raw_winding - integer_winding);
    if is_resolved && winding_error <= 0.20 && abs(integer_winding) >= 1
        circulation = 2 * pi * isf.hbar * integer_winding;
    else
        circulation = NaN;
    end
end

function index = coordinate_to_index(coordinate, spacing, resolution)
    index = mod(round(coordinate / spacing), resolution) + 1;
end

function indices = wrap_indices(first, count, resolution)
    indices = mod((first - 1) + (0:count - 1), resolution) + 1;
end

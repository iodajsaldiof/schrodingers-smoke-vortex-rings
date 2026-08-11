function cycle = detect_cycle(track_data, x_period, axial_spacing)
%DETECT_CYCLE 以连续展开轨迹和可信帧判定交替穿越。
%
% 真实穿越必须在相互接近前后都有至少两个“可分辨的双环”观测点，且相对
% 轴向位置的符号改变。涡核合并时的峰值跳变会被记录为 ambiguity，而不是
% 被错误计作穿越。这比对每一帧的两个局部极大值直接数符号更严格。

    [time, x1, x2, resolved] = extract_track_arrays(track_data);
    cycle = empty_cycle();
    if numel(time) < 5
        return
    end
    finite = isfinite(time) & isfinite(x1) & isfinite(x2);
    time = time(finite);
    x1 = x1(finite);
    x2 = x2(finite);
    resolved = resolved(finite);
    if numel(time) < 5
        return
    end

    difference = x1 - x2;
    % 若旧数据没有连续展开列，使用周期最短差作为兼容后备。
    if max(abs(difference)) > 1.5 * x_period
        difference = wrap_signed(difference, x_period);
    end
    tolerance = max(1.5 * axial_spacing, 0.01 * x_period);
    stable = resolved & abs(difference) >= tolerance;
    cycle.unresolved_fraction = 1.0 - mean(resolved);
    cycle.axial_overlap_fraction = 1.0 - mean(abs(difference) >= tolerance);
    stable_indices = find(stable);
    if numel(stable_indices) < 4
        return
    end
    signs = sign(difference(stable_indices));
    candidate_times = zeros(0, 1);
    confirmed_times = zeros(0, 1);
    for index = 2:numel(stable_indices)
        if signs(index) == signs(index - 1)
            continue
        end
        left_index = stable_indices(index - 1);
        right_index = stable_indices(index);
        candidate_times(end + 1, 1) = interpolate_zero( ...
            time(left_index), difference(left_index), time(right_index), difference(right_index)); %#ok<AGROW>

        left_run = same_sign_run(signs, index - 1, -1);
        right_run = same_sign_run(signs, index, 1);
        if left_run >= 2 && right_run >= 2
            confirmed_times(end + 1, 1) = candidate_times(end); %#ok<AGROW>
        end
    end
    cycle.candidate_exchange_count = numel(candidate_times);
    cycle.exchange_count = numel(confirmed_times);
    if ~isempty(confirmed_times)
        cycle.first_exchange_time = confirmed_times(1);
    end
    if numel(confirmed_times) >= 2
        cycle.second_exchange_time = confirmed_times(2);
        cycle.order_return_time = confirmed_times(2);
        cycle.order_cycle_detected = true;
        cycle.mean_exchange_interval = mean(diff(confirmed_times));
    end
end

function cycle = empty_cycle()
    cycle = struct('candidate_exchange_count', 0, 'exchange_count', 0, ...
        'first_exchange_time', NaN, 'second_exchange_time', NaN, ...
        'order_return_time', NaN, 'order_cycle_detected', false, ...
        'mean_exchange_interval', NaN, 'unresolved_fraction', NaN, ...
        'axial_overlap_fraction', NaN);
end

function [time, x1, x2, resolved] = extract_track_arrays(track_data)
    if istable(track_data)
        time = track_data.time;
        if ismember('ring_1_x_unwrapped', track_data.Properties.VariableNames)
            x1 = track_data.ring_1_x_unwrapped;
            x2 = track_data.ring_2_x_unwrapped;
        else
            x1 = track_data.ring_1_x;
            x2 = track_data.ring_2_x;
        end
        if ismember('track_resolved', track_data.Properties.VariableNames)
            resolved = logical(track_data.track_resolved);
        elseif ismember('track_valid', track_data.Properties.VariableNames)
            resolved = logical(track_data.track_valid);
        else
            resolved = true(size(time));
        end
    else
        time = track_data(:, 2);
        x1 = track_data(:, 3);
        x2 = track_data(:, 5);
        if size(track_data, 2) >= 18
            resolved = logical(track_data(:, 18));
        else
            resolved = true(size(time));
        end
    end
    time = double(time);
    x1 = double(x1);
    x2 = double(x2);
end

function count = same_sign_run(signs, start_index, increment)
    count = 1;
    index = start_index + increment;
    while index >= 1 && index <= numel(signs) && signs(index) == signs(start_index)
        count = count + 1;
        index = index + increment;
    end
end

function time = interpolate_zero(t0, d0, t1, d1)
    fraction = abs(d0) / (abs(d0) + abs(d1));
    time = t0 + fraction * (t1 - t0);
end

function difference = wrap_signed(value, period)
    difference = mod(value + 0.5 * period, period) - 0.5 * period;
end

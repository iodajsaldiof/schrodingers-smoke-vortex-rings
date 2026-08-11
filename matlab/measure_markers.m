function values = measure_markers(markers, volume_size)
%MEASURE_MARKERS 输出带身份的材料环心、平均半径与形变离散度。
%
% x 坐标保持粒子推进给出的连续坐标，不对 x 取模；这消除了周期边界造成的
% 轨迹跳变。半径在 y-z 周期截面中测量，标准差用于标记环是否严重拉伸失真。

    values = struct('x1', NaN, 'radius1', NaN, 'radius_std1', NaN, ...
        'x2', NaN, 'radius2', NaN, 'radius_std2', NaN, 'resolved', false);
    radii = cell(1, 2);
    for ring_index = 1:2
        marker = markers{ring_index};
        y_offset = wrap_signed(marker.y - 0.5 * volume_size(2), volume_size(2));
        z_offset = wrap_signed(marker.z - 0.5 * volume_size(3), volume_size(3));
        radii{ring_index} = sqrt(y_offset.^2 + z_offset.^2);
        if ring_index == 1
            values.x1 = mean(marker.x);
            values.radius1 = mean(radii{ring_index});
            values.radius_std1 = std(radii{ring_index});
        else
            values.x2 = mean(marker.x);
            values.radius2 = mean(radii{ring_index});
            values.radius_std2 = std(radii{ring_index});
        end
    end
    % 标记环半径标准差不超过平均半径的 40% 时，中心仍可代表一个连贯涡环。
    values.resolved = values.radius_std1 <= 0.40 * values.radius1 && ...
        values.radius_std2 <= 0.40 * values.radius2;
end

function value = wrap_signed(value, period)
    value = mod(value + 0.5 * period, period) - 0.5 * period;
end

function markers = initialize_markers(config)
%INITIALIZE_MARKERS 在两条初始涡丝上分别播撒带身份的材料标记。
%
% 论文 5 的 example_leapfrog.m 已使用 Particles 进行拉格朗日可视化。这里将
% 其推广为两组闭合标记环：每组从对应初始涡丝出发，因而能跨越几何峰值合并阶段
% 保持“最初是哪一条环”的材料身份。

    marker_count = config.material_marker_count;
    angles = (0:marker_count - 1).' * (2 * pi / marker_count);
    markers = cell(1, 2);
    for ring_index = 1:2
        marker = Particles;
        center = config.ring_centers(ring_index, :);
        radius = config.ring_radii(ring_index);
        marker.x = center(1) + zeros(marker_count, 1);
        marker.y = center(2) + radius * cos(angles);
        marker.z = center(3) + radius * sin(angles);
        markers{ring_index} = marker;
    end
end

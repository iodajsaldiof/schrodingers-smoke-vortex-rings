function plot_diagnostics(output_path, tracking, material_tracking, diagnostics, cycle)
%PLOT_DIAGNOSTICS 输出轨迹、可信度、守恒量与周期判据图。
%
% 轴向轨迹采用跨周期边界连续展开的 x，而不是盒内 [0,Lx) 坐标；灰色叉号
% 表示两环涡核合并、身份不可可靠区分的帧，不能用来证明穿越。

    figure_handle = figure('Visible', 'off', 'Color', 'w', ...
        'Position', [80, 80, 1180, 900]);
    layout = tiledlayout(2, 2, 'TileSpacing', 'loose', 'Padding', 'loose');
    % ring_tracks.csv 保留欧拉峰值数据；B 题的主轨迹使用材料标记，从而能跨越
    % 涡环超越事件保持最初的编号。
    resolved = logical(material_tracking.track_resolved);
    ambiguous = ~resolved;

    nexttile(layout);
    hold on;
    first_handle = plot(material_tracking.time, material_tracking.ring_1_x_unwrapped, 'LineWidth', 1.4);
    second_handle = plot(material_tracking.time, material_tracking.ring_2_x_unwrapped, 'LineWidth', 1.4);
    legend_handles = [first_handle, second_handle];
    legend_labels = {'ring 1', 'ring 2'};
    if any(ambiguous)
        ambiguous_handle = scatter(material_tracking.time(ambiguous), material_tracking.ring_1_x_unwrapped(ambiguous), ...
            16, [0.35, 0.35, 0.35], 'x');
        scatter(material_tracking.time(ambiguous), material_tracking.ring_2_x_unwrapped(ambiguous), ...
            16, [0.35, 0.35, 0.35], 'x');
        legend_handles(end + 1) = ambiguous_handle;
        legend_labels{end + 1} = 'ambiguous extraction';
    end
    mark_exchange_times(cycle);
    hold off; grid on;
    xlabel('time'); ylabel('material axial center x');
    legend(legend_handles, legend_labels, 'Location', 'best');
    title('Material-marker axial center trajectories');

    nexttile(layout);
    hold on;
    first_handle = plot(material_tracking.time, material_tracking.ring_1_material_radius, 'LineWidth', 1.4);
    second_handle = plot(material_tracking.time, material_tracking.ring_2_material_radius, 'LineWidth', 1.4);
    legend_handles = [first_handle, second_handle];
    legend_labels = {'ring 1', 'ring 2'};
    if any(ambiguous)
        ambiguous_handle = scatter(material_tracking.time(ambiguous), material_tracking.ring_1_material_radius(ambiguous), ...
            16, [0.35, 0.35, 0.35], 'x');
        scatter(material_tracking.time(ambiguous), material_tracking.ring_2_material_radius(ambiguous), ...
            16, [0.35, 0.35, 0.35], 'x');
        legend_handles(end + 1) = ambiguous_handle;
        legend_labels{end + 1} = 'ambiguous extraction';
    end
    hold off; grid on;
    xlabel('time'); ylabel('extracted radius');
    legend(legend_handles, legend_labels, 'Location', 'best');
    title('Material-marker radii and coherence');

    nexttile(layout);
    semilogy(diagnostics.time, diagnostics.edge_divergence_l2, 'LineWidth', 1.3); hold on;
    semilogy(diagnostics.time, diagnostics.density_constraint_l2, 'LineWidth', 1.3); hold off; grid on;
    xlabel('time'); ylabel('L2 residual');
    legend({'divergence', 'density constraint'}, 'Location', 'best');
    title('Constraint residuals');

    nexttile(layout);
    yyaxis left;
    plot(diagnostics.time, diagnostics.total_energy, 'LineWidth', 1.4); hold on;
    plot(diagnostics.time, diagnostics.kinetic_energy, 'LineWidth', 1.4); hold off;
    ylabel('energy');
    yyaxis right;
    plot(tracking.time, tracking.ring_1_circulation, '--', 'LineWidth', 1.2); hold on;
    plot(tracking.time, tracking.ring_2_circulation, '--', 'LineWidth', 1.2); hold off;
    ylabel('quantized circulation'); grid on;
    xlabel('time');
    title('Energy and reliable local circulation');

    title(layout, sprintf('ISF diagnostics: %d confirmed exchanges, %.1f%% unresolved', ...
        cycle.exchange_count, 100 * cycle.unresolved_fraction));
    print(figure_handle, fullfile(output_path, 'diagnostics.png'), '-dpng', '-r220');
    close(figure_handle);
end

function mark_exchange_times(cycle)
    if isfinite(cycle.first_exchange_time)
        xline(cycle.first_exchange_time, '--k', 'first confirmed exchange');
    end
    if isfinite(cycle.second_exchange_time)
        xline(cycle.second_exchange_time, '--k', 'order return');
    end
end

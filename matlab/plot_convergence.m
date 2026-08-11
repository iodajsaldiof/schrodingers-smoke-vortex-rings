function plot_convergence(summary, output_root)
%PLOT_CONVERGENCE 绘制并保存收敛性汇总图。
%
% 单独写成函数后，已有的 convergence_summary.csv 可直接重画，
% 不必重新运行四组耗时的数值试验。

    required_names = ["space_coarse", "baseline", "space_fine", "time_fine"];
    order = categorical(string(summary.case_name), required_names, 'Ordinal', true);

    figure_handle = figure('Visible', 'off', 'Color', 'w', ...
        'Position', [80, 80, 980, 430]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    bar(order, summary.order_return_time, 'FaceColor', [0.10, 0.45, 0.72]);
    grid on;
    ylabel('order-return time');
    title('Leapfrogging timing convergence');
    ax = gca;
    ax.TickLabelInterpreter = 'none';

    nexttile;
    semilogy(order, summary.divergence_l2_at_stop, 'o-', ...
        'Color', [0.00, 0.45, 0.70], 'LineWidth', 1.5, ...
        'MarkerSize', 7, 'MarkerFaceColor', 'w');
    grid on;
    ylabel('L2 divergence');
    title('Constraint residual');
    ax = gca;
    ax.TickLabelInterpreter = 'none';

    print(figure_handle, fullfile(output_root, 'convergence_summary.png'), '-dpng', '-r220');
    close(figure_handle);
end

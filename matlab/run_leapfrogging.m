function result = run_leapfrogging(output_path, max_steps)
%RUN_PAPER5_LEAPFROG 复现 Chern et al. Figure 4 的双涡环交替穿越基准。
%
% 该算例使用作者公开 example_leapfrog.m 的网格、时间步、背景速度及
% R=[1.5,0.9] 的共心双环。论文报告该设置在 2000 步后仍保持约四个周期；
% 本项目额外输出可复核的环心轨迹、可信度和穿越判据。

    if nargin < 1
        output_path = '';
    end
    if nargin < 2
        max_steps = 2000;
    end
    config = make_config('leapfrogging', output_path, max_steps);
    result = run_simulation(config);
end

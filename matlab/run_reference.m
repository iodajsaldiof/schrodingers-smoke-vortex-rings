function result = run_reference(output_path, max_steps)
%RUN_REFERENCE 运行 Chern et al. example_leapfrog 的可诊断复现。
%
% 此函数仍使用作者示例的共心、异半径涡环参数，但现在也会输出环心轨迹、
% 半径、局部环量、能量和顺序交换判据，便于同 B 题工况逐项比较。

    if nargin < 1
        output_path = '';
    end
    if nargin < 2
        max_steps = 2000;
    end
    config = make_config('reference', output_path, max_steps);
    result = run_simulation(config);
end

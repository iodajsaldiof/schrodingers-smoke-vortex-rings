function result = run_equal_rings(output_path, max_steps)
%RUN_EQUAL_RINGS 运行等半径同轴双涡环的推荐对照工况。
%
% 默认采用 R=1、d=0.6、128x64x64 网格和 dt=1/24。程序检测到两次轴向
% 顺序交换后会停止；若在 max_steps 内未检测到，则保留全部数据并明确报告。
%
% 示例：
%   result = run_equal_rings
%   result = run_equal_rings('D:/.../outputs/equal_rings_run', 2400)

    if nargin < 1
        output_path = '';
    end
    if nargin < 2
        max_steps = 2000;
    end
    config = make_config('equal_rings', output_path, max_steps);
    result = run_simulation(config);
end

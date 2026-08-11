function values = compute_diagnostics(isf, psi1, psi2)
%COMPUTE_DIAGNOSTICS 计算 ISF 的不可压、能量和归一化诊断量。
%
% total_energy 使用 Chern et al. 能量表述中的两分量波函数 Dirichlet 能；
% kinetic_energy 则由物理速度 u=hbar*grad(phase) 得到，二者同时保存，避免
% 把波函数正则化能和经典动能混为同一个量。

    [vx, vy, vz] = isf.VelocityOneForm(psi1, psi2, isf.hbar);
    divergence = isf.Div(vx, vy, vz);
    [ux, uy, uz] = isf.Sharp(vx, vy, vz);
    cell_volume = isf.dx * isf.dy * isf.dz;

    values.edge_divergence_l2 = sqrt(mean(abs(divergence(:)).^2));
    values.kinetic_energy = 0.5 * cell_volume * sum( ...
        abs(ux(:)).^2 + abs(uy(:)).^2 + abs(uz(:)).^2);

    dpsi1_x = (circshift(psi1, [-1, 0, 0]) - psi1) / isf.dx;
    dpsi1_y = (circshift(psi1, [0, -1, 0]) - psi1) / isf.dy;
    dpsi1_z = (circshift(psi1, [0, 0, -1]) - psi1) / isf.dz;
    dpsi2_x = (circshift(psi2, [-1, 0, 0]) - psi2) / isf.dx;
    dpsi2_y = (circshift(psi2, [0, -1, 0]) - psi2) / isf.dy;
    dpsi2_z = (circshift(psi2, [0, 0, -1]) - psi2) / isf.dz;
    gradient_density = abs(dpsi1_x).^2 + abs(dpsi1_y).^2 + abs(dpsi1_z).^2 + ...
        abs(dpsi2_x).^2 + abs(dpsi2_y).^2 + abs(dpsi2_z).^2;
    values.total_energy = 0.5 * isf.hbar^2 * cell_volume * sum(gradient_density(:));

    density = abs(psi1).^2 + abs(psi2).^2;
    values.density_constraint_l2 = sqrt(mean((density(:) - 1.0).^2));
end

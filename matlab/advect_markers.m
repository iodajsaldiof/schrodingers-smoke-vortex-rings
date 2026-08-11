function markers = advect_markers(isf, psi1, psi2, markers)
%ADVECT_MARKERS 使用作者的 RK4 粒子推进保持两组材料标签。
    [vx, vy, vz] = isf.VelocityOneForm(psi1, psi2, isf.hbar);
    [vx, vy, vz] = isf.StaggeredSharp(vx, vy, vz);
    for ring_index = 1:numel(markers)
        markers{ring_index}.StaggeredAdvect(isf, vx, vy, vz, isf.dt);
    end
end

%% Bayesian history matching on a nonlinear 2D oil-water model (synthetic twin experiment)
% This script reproduces a synthetic two-phase data assimilation setup with
% MAP, RML, and MCMC posterior inference for log-permeability.
%
% Key setup requested:
%   * 15 x 30 Cartesian grid
%   * 6 producers + 2 injectors in a five-spot inspired pattern
%   * Gaussian log-perm prior with mean 5, variance 1
%   * Rotated anisotropic exponential covariance (major=60, minor=5 cells, 45 deg)
%   * 3900 days history (150 day spacing), plus 3900 days forecast
%   * 5% iid noise in oil/water-rate observations
%   * MAP via quasi-Newton minimization of Bayesian objective
%   * 100 prior samples transformed to posterior via RML
%   * Comparison against MCMC posterior samples
%
% Notes:
%   1) The full experiment is computationally heavy. Use the options in
%      `cfg` to reduce workload for quick checks.
%   2) Gradient in MAP is computed numerically by default. If you have a
%      custom adjoint gradient implementation wrt rock-perm, plug it into
%      `objectiveAndGradient`.
%
% MRST modules needed: ad-core, ad-props, ad-blackoil

mrstModule add ad-core ad-props ad-blackoil
rng(20260415, 'twister');

%% Configuration
cfg = struct();
cfg.nx = 15;
cfg.ny = 30;
cfg.Lx = 1500*meter;
cfg.Ly = 3000*meter;
cfg.poro = 0.2;
cfg.logkMean = 5.0;
cfg.logkVar = 1.0;
cfg.corrMajor = 60;   % cells
cfg.corrMinor = 5;    % cells
cfg.thetaDeg = 45;    % rotation
cfg.sigmaObsFrac = 0.05;
cfg.dt = 150*day;
cfg.tHist = 3900*day;
cfg.tPred = 3900*day;
cfg.nRML = 100;
cfg.nMCMC = 300;      % kept moderate; increase for publication-grade
cfg.burnIn = 100;
cfg.stepScale = 0.08; % random-walk scale in reduced coordinates
cfg.nKL = 40;         % reduced dimensions for MCMC proposals
cfg.maxMapIter = 35;

%% Grid and covariance
[G, idx] = makeGrid(cfg);
N = G.cells.num;
mu = cfg.logkMean*ones(N,1);
[Cm, Lc] = buildPriorCovariance(G, cfg, idx);

%% True realization and synthetic observations
mTrue = mu + Lc*randn(N,1);
[modelTrue, state0, schedule, histIx, predIx] = buildSimulationModel(G, exp(mTrue), cfg, idx);
[wellSolsTrue, statesTrue] = simulateScheduleAD(state0, modelTrue, schedule);
[dObs, Cd] = makeObservations(wellSolsTrue, histIx, cfg);

%% MAP (quasi-Newton)
m0 = mu;
objFun = @(m) objectiveOnly(m, G, cfg, idx, schedule, state0, dObs, Cd, mu, Cm, histIx);
opt = optimoptions('fminunc', ...
    'Algorithm', 'quasi-newton', ...
    'Display', 'iter', ...
    'MaxIterations', cfg.maxMapIter, ...
    'MaxFunctionEvaluations', 500);
[mMAP, fvalMAP] = fminunc(objFun, m0, opt); %#ok<ASGLU>

%% RML posterior ensemble
mPriorEns = mu + Lc*randn(N, cfg.nRML);
mRMLEns = zeros(N, cfg.nRML);
for i = 1:cfg.nRML
    dPert = dObs + sqrt(diag(Cd)).*randn(size(dObs));
    objRML = @(m) objectiveOnly(m, G, cfg, idx, schedule, state0, dPert, Cd, mPriorEns(:,i), Cm, histIx);
    mRMLEns(:,i) = fminunc(objRML, mPriorEns(:,i), opt);
    fprintf('RML member %d/%d done\n', i, cfg.nRML);
end

%% MCMC (random walk in reduced KL space)
[Vkl, Dkl] = eigs(Cm, cfg.nKL, 'largestabs');
Skl = sqrt(max(diag(Dkl), 0));
z0 = zeros(cfg.nKL,1);
logp = @(z) -objectiveOnly(mu + Vkl*(Skl.*z), G, cfg, idx, schedule, state0, dObs, Cd, mu, Cm, histIx);
[zChain, accRate] = runMCMC(logp, z0, cfg.nMCMC, cfg.stepScale);
fprintf('MCMC acceptance rate: %.2f\n', accRate);
zKeep = zChain(:, cfg.burnIn+1:end);
nKeep = size(zKeep,2);
mMCMCEns = mu + Vkl*(Skl.*zKeep);

%% Forecast comparison
[~, qTrueHist, qTruePred] = evaluateQoI(mTrue, G, cfg, idx, schedule, state0, histIx, predIx);
[~, qMapHist, qMapPred] = evaluateQoI(mMAP,  G, cfg, idx, schedule, state0, histIx, predIx);

qRMLPred = zeros(numel(qTruePred), cfg.nRML);
for i = 1:cfg.nRML
    [~, ~, qRMLPred(:,i)] = evaluateQoI(mRMLEns(:,i), G, cfg, idx, schedule, state0, histIx, predIx);
end
qMCMCPred = zeros(numel(qTruePred), nKeep);
for i = 1:nKeep
    [~, ~, qMCMCPred(:,i)] = evaluateQoI(mMCMCEns(:,i), G, cfg, idx, schedule, state0, histIx, predIx);
end

%% Visualization
figure('Name', 'Permeability recovery', 'Position', [100,100,1400,420]);
subplot(1,3,1); plotCellData(G, mTrue); axis equal tight; title('True log-perm'); colorbar
subplot(1,3,2); plotCellData(G, mMAP); axis equal tight; title('MAP log-perm'); colorbar
subplot(1,3,3); plotCellData(G, mean(mRMLEns,2)); axis equal tight; title('RML mean log-perm'); colorbar

figure('Name', 'Forecast uncertainty', 'Position', [100,100,1300,450]);
plotForecastBands(qTruePred, qMapPred, qRMLPred, qMCMCPred);

disp('Experiment complete.');

%% ----------------------------- helpers ---------------------------------
function [G, idx] = makeGrid(cfg)
    G = computeGeometry(cartGrid([cfg.nx, cfg.ny, 1], [cfg.Lx, cfg.Ly, 10*meter]));
    [ix, iy] = ndgrid(1:cfg.nx, 1:cfg.ny);
    idx = [ix(:), iy(:)];
end

function [Cm, Lc] = buildPriorCovariance(G, cfg, idx)
    N = G.cells.num;
    C = zeros(N,N);
    th = deg2rad(cfg.thetaDeg);
    R = [cos(th), -sin(th); sin(th), cos(th)];
    for i = 1:N
        for j = i:N
            d = idx(i,:) - idx(j,:);
            dr = (R'*d(:))';
            h = sqrt((dr(1)/cfg.corrMajor)^2 + (dr(2)/cfg.corrMinor)^2);
            cij = cfg.logkVar*exp(-h);
            C(i,j) = cij;
            C(j,i) = cij;
        end
    end
    C = C + 1e-8*eye(N);
    Cm = C;
    Lc = chol(Cm, 'lower');
end

function [model, state0, schedule, histIx, predIx] = buildSimulationModel(G, kMilliDarcy, cfg, idx)
    rock = makeRock(G, kMilliDarcy(:)*milli*darcy, cfg.poro);
    fluid = initSimpleADIFluid('phases', 'WO', ...
        'mu', [1, 2]*centi*poise, ...
        'rho', [1000, 700]*kilogram/meter^3, ...
        'n', [2,2]);
    model = TwoPhaseOilWaterModel(G, rock, fluid);

    W = [];
    pCells = getWellCells(cfg.nx, cfg.ny, idx);
    qI = 500*meter^3/day;
    qP = -qI*(2/6);

    for k = 1:2
        W = addWell(W, G, rock, pCells.inj(k), ...
            'Type', 'rate', 'Val', qI, 'Comp_i', [1, 0], ...
            'Name', sprintf('I%d', k));
    end
    for k = 1:6
        W = addWell(W, G, rock, pCells.prod(k), ...
            'Type', 'rate', 'Val', qP, 'Comp_i', [0, 1], ...
            'Name', sprintf('P%d', k));
    end

    nHist = round(cfg.tHist/cfg.dt);
    nPred = round(cfg.tPred/cfg.dt);
    nTotal = nHist + nPred;
    schedule = simpleSchedule(repmat(cfg.dt, nTotal, 1), 'W', W);
    histIx = 1:nHist;
    predIx = (nHist+1):nTotal;

    state0 = initResSol(G, 250*barsa, [0.15, 0.85]);
end

function cells = getWellCells(nx, ny, idx)
    picksInj = [2, 2; nx-1, ny-1];
    picksProd = [2, ny-1; nx-1, 2; round(nx/2), 2; round(nx/2), ny-1; 2, round(ny/2); nx-1, round(ny/2)];
    cells.inj = zeros(size(picksInj,1),1);
    cells.prod = zeros(size(picksProd,1),1);
    for i = 1:numel(cells.inj)
        cells.inj(i) = find(idx(:,1)==picksInj(i,1) & idx(:,2)==picksInj(i,2), 1);
    end
    for i = 1:numel(cells.prod)
        cells.prod(i) = find(idx(:,1)==picksProd(i,1) & idx(:,2)==picksProd(i,2), 1);
    end
end

function [dObs, Cd] = makeObservations(wellSols, histIx, cfg)
    data = extractProdData(wellSols, histIx);
    sigma = cfg.sigmaObsFrac*max(abs(data), 1e-8);
    dObs = data + sigma.*randn(size(data));
    Cd = diag(sigma.^2);
end

function d = extractProdData(wellSols, tIx)
    d = [];
    for t = tIx
        ws = wellSols{t};
        for w = 1:numel(ws)
            if ws(w).qWs < 0 || ws(w).qOs < 0
                d = [d; ws(w).qWs; ws(w).qOs]; %#ok<AGROW>
            end
        end
    end
end

function J = objectiveOnly(m, G, cfg, idx, schedule, state0, dObs, Cd, mref, Cm, histIx)
    [model, ~, ~, ~, ~] = buildSimulationModel(G, exp(m), cfg, idx);
    wellSols = simulateScheduleAD(state0, model, schedule);
    dPred = extractProdData(wellSols, histIx);
    dm = m - mref;
    dd = dObs - dPred;
    Jm = 0.5*(dm'*(Cm\dm));
    Jd = 0.5*(dd'*(Cd\dd));
    J = Jm + Jd;
end

function [wellSols, qHist, qPred] = evaluateQoI(m, G, cfg, idx, schedule, state0, histIx, predIx)
    [model, ~, ~, ~, ~] = buildSimulationModel(G, exp(m), cfg, idx);
    wellSols = simulateScheduleAD(state0, model, schedule);
    qHist = extractProdData(wellSols, histIx);
    qPred = extractProdData(wellSols, predIx);
end

function [chain, accRate] = runMCMC(logp, z0, nstep, stepScale)
    n = numel(z0);
    chain = zeros(n, nstep);
    z = z0;
    lp = logp(z);
    acc = 0;
    for i = 1:nstep
        zp = z + stepScale*randn(n,1);
        lpp = logp(zp);
        if log(rand) < (lpp - lp)
            z = zp;
            lp = lpp;
            acc = acc + 1;
        end
        chain(:,i) = z;
    end
    accRate = acc/nstep;
end

function plotForecastBands(qTrue, qMap, qRML, qMCMC)
    n = numel(qTrue);
    t = 1:n;
    prcR = prctile(qRML', [5,50,95])';
    prcM = prctile(qMCMC', [5,50,95])';

    hold on
    fill([t, fliplr(t)], [prcR(:,1)', fliplr(prcR(:,3)')], [0.7, 0.85, 1], ...
        'EdgeColor', 'none', 'FaceAlpha', 0.5);
    fill([t, fliplr(t)], [prcM(:,1)', fliplr(prcM(:,3)')], [1, 0.85, 0.7], ...
        'EdgeColor', 'none', 'FaceAlpha', 0.35);
    plot(t, qTrue, 'k-', 'LineWidth', 1.8);
    plot(t, qMap, 'b--', 'LineWidth', 1.5);
    plot(t, prcR(:,2), '-', 'Color', [0.1, 0.4, 0.9], 'LineWidth', 1.1);
    plot(t, prcM(:,2), '-', 'Color', [0.9, 0.4, 0.1], 'LineWidth', 1.1);
    legend({'RML 90% band', 'MCMC 90% band', 'True', 'MAP', 'RML median', 'MCMC median'}, 'Location', 'best');
    xlabel('Forecast data index');
    ylabel('Rate (SI)');
    title('MAP point forecast vs posterior predictive uncertainty');
    grid on
    hold off
end

function [logLiklVals] = lrgLogLiklWrap(evalAtPts)
% evalPts is an numPtsx9 array. Each row is a point at which we want to
% evaluate the log Likelihood.
% These are the cosmological Parameters
% Omega_k     : Spatial Curvature [-1, 0.9]
% Omega_Lambda: Dark Energy Fraction [0, 1]
% omega_C     : Cold Dark Matter Density [0 1.2]
% omega_B     : Baryon Density [0.001 - 0.25]
% n_s         : Scalar Spectral Index [0.5 - 1.7]
% A_s         : Scalar Fluctuation Amplitude = 0.6845
% alpha       : Running of Spectral Index = 0.0
% b           : Galaxy Bias [0.0 3.0]
% Q_nl        : Nonlinear Correction = 30.81

  LOWESTLOGLIKLVAL = -10000;

  numPts = size(evalAtPts, 1);
  numDims = size(evalAtPts, 2); % This should be 9 ?
  fortOutFile = sprintf('lOut_%s_%d.txt', datestr(now, 'HHMMSS'), randi(9999999) );
  binName = 'bings13.out';
  outFile = sprintf('sim/%s', fortOutFile);

  logLiklVals = zeros(numPts, 1);

  % NOw call the simulator iteratively
  for iter = 1:numPts

    % First create the command
    currEvalPt = evalAtPts(iter, :);
    commandStr = sprintf('cd sim && ./%s ', binName);
    for k = 1:9
      commandStr = sprintf('%s %f ', commandStr, currEvalPt(k));
    end
    commandStr = sprintf('%s %s && cd ..', commandStr, fortOutFile);
    
    % Execute the command
    system(commandStr);

    % Read from file
    outVal = load(outFile);
    if isnan(outVal) outVal = LOWESTLOGLIKLVAL; end
%     if isnan(outVal) | outVal < LOWESTLOGLIKLVAL, outVal = LOWESTLOGLIKLVAL; end
    outVal = outVal;
%     outVal = log(-LOWESTLOGLIKLVAL + 1 + outVal);
    logLiklVals(iter) = outVal;

  end

  % Delete the file
  delete(outFile);

end


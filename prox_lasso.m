function [beta, fhist] = prox_lasso(X, y, lambda, iters, t0)
    if nargin < 3 || isempty(lambda), lambda = 1; end
    if nargin < 4 || isempty(iters), iters = 5000; end
    if nargin < 5 || isempty(t0), t0 = 1 / norm(X)^2; end

    [n, p] = size(X);
    y = y(:);

    f = @(b) 0.5*(y - X*b)'*(y - X*b) + lambda * sum(abs(b));
    fhist = zeros(2, iters, 1);

    % Proximal Descent
    beta = zeros(p, 1);

    for k = 1:iters
        % Differential Gradient
        t = t0;
        g = beta + t*X'*(y - X*beta);

        % Closed Form Solution
        beta = sign(g) .* max(abs(g) - lambda*t, 0);

        fhist(1, k) = f(beta);
    end

    % Subgradient Method
    beta = zeros(p, 1);

    for k = 1:iters
      t = t0 / sqrt(k);
      g = -X'*(y - X*beta) + lambda*sign(beta);

      beta = beta - t*g;

      fhist(2, k) = f(beta);
    end


%    % Stochastic Subgradient Method
%    beta = zeros(p, 1);
%
%    for k = 1:iters
%      i = randi(n-5);
%      t = t0 / sqrt(k);
%      g = -(n-5) * X(i:i+5,:)'*(y(i:i+5) - X(i:i+5,:)*beta) + lambda*sign(beta);
%
%      beta = beta - t*g;
%
%      fhist(3, k) = f(beta);
%    end
end

% Synthetic data
%clear;
%n = 200; p = 2;
%X = [randn(n/2,p) + 2; randn(n/2,p) - 2];
%y = [ones(n/2,1); -ones(n/2,1)];
%[beta, fhist] = prox_lasso(X, y, 1, 5e+2, 1 / norm(X)^2);
%title("Synthetic: Lasso Least Squares, Proximal Descent"); xlabel('k'); ylabel('f');
%plot(fhist(1, :) + eps, 'color', 'b'); hold on;
%plot(fhist(2, :) + eps, 'color', 'm');
%plot(fhist(3, :) + eps, 'color', 'k');
%legend("Prox", "Subgrad", "Stochastic Subgrad"); hold off;

% w3a data
clear;
[y, X] = load_libsvm('w3a', 300);  % 300 = number of features in w3a
[beta, fhist] = prox_lasso(X, y, 1, 5e+2, 1 / norm(X)^2);
title("w3a: Lasso Least Squares, Proximal Descent"); xlabel('k'); ylabel('f');
plot(fhist(1, :) + eps, 'color', 'b'); hold on;
plot(fhist(2, :) + eps, 'color', 'm');
legend("Prox", "Subgrad"); hold off;

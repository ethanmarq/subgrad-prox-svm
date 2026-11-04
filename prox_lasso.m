function [beta, fhist] = prox_lasso(X, y, lambda, iters, t0)
    if nargin < 3 || isempty(lambda), lambda = 1; end
    if nargin < 4 || isempty(iters), iters = 5000; end
    if nargin < 5 || isempty(t0), t0 = 1e-2; end

    % Number of Step comparsions!
    N = 8;

    [n, p] = size(X);
    y = y(:);

    % f = @(b, c) 0.5*(b'*b) + C*sum(max(0, 1 - y.*(X*b + c)));
    f = @(b) 0.5*(y - X*b)'(y - X*b) + lambda * sum(abs(b));

    % fhist = zeros(N, iters, 1);
    fhist = zeros(iters, 1);

    % for i = 1:N
    beta = zeros(p, 1);

    for k = 1:iters
        % steps = [5e-1, 1, k^(1/5), k^(1/4), k^(1/3), sqrt(k), k, -1];

        % Differential Gradient
        t = t0;
        g =  beta - t*X'*(y - X*beta);

        beta = soft_thres(g, lambda, t);


        % Track Iterate
        fhist(k) = f(beta);

        % end
    end
end

function [beta] = soft_thres(beta, lambda, t)
    if (beta > lambda*t)
        beta = beta - lambda*t;
    else if (beta < -lambda*t)
        beta = beta + lambda*t;
    else if (abs(beta) <= lambda*t)
        beta = beta;
    end
end

% Synthetic data
clear;
n = 200; p = 2;
X = [randn(n/2,p) + 2; randn(n/2,p) - 2];
y = [ones(n/2,1); -ones(n/2,1)];
[beta, fhist] = prox_lasso(X, y, 1, 5e+2, 1);
hold on;
title("Synthetic: Lasso Least Squares, Proximal Descent");

% w3a data
clear;
[y, X] = load_libsvm('w3a', 300);  % 300 = number of features in w3a
[beta, fhist] = prox_lasso(X, y, 1, 5e+2, 1);
hold on;
title("w3a: Lasso Least Squares, Proximal Descent");

% Plotting
plot(fhist(1) + eps, 'color', 'b'); xlabel('k'); ylabel('f'); hold off;



% Current Dead Code
plot(fhist(1,:) + eps, 'color', 'b'); xlabel('k'); ylabel('f');
plot(fhist(2,:) + eps, 'color', 'm');
plot(fhist(3,:) + eps, 'color', 'k');
plot(fhist(4,:) + eps, 'color', 'c');
plot(fhist(5,:) + eps, 'color', 'g');
plot(fhist(6,:) + eps, 'color', [1 0.5 0]);
plot(fhist(7,:) + eps, 'color', [0 0.5 1]);
plot(cummin(fhist(8,:)) + eps, 'color', 'r'); hold off;
legend("5e-1", "1", "k^{1/5}", "k^{1/4}", "k^{1/3}", "k^{1/2}", "k", "Polyak");
hold off;

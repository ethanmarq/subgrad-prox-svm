function [beta, b0, fhist] = svm_subgrad(X, y, C, iters, t0)
    if nargin < 3 || isempty(C), C = 1; end
    if nargin < 4 || isempty(iters), iters = 5000; end
    if nargin < 5 || isempty(t0), t0 = 1e-2; end

    % N = length(steps)
    N = 8;
    [n, p] = size(X);
    y = y(:);

    f = @(b, c) 0.5*(b'*b) + C*sum(max(0, 1 - y.*(X*b + c)));
    fhist = zeros(N, iters, 1);

    % Step Comparison
    for i = 1:N
        beta = zeros(p, 1);
        b0 = 0;

        % Subgradient Method
        for k = 1:iters
            steps = [5e-1, 1, k^(1/5), k^(1/4), k^(1/3), sqrt(k), k, -1];

            act = (y .* (X*beta + b0)) < 1; % margin violators I = {i : y_i (x_i'beta + b0) < 1}

            g_beta = beta - C * (X(act,:)' * y(act));
            g_b0 = -C * sum(y(act));

            if (steps(i) == -1)
                % Polyak Step
                t = f(beta, b0) / (norm([g_beta; g_b0], 2)^2 + eps);
            else
                % diminishing step
                t = t0 / steps(i);
            end

            beta = beta - t*g_beta;
            b0 = b0 - t*g_b0;

            fhist(i, k) = f(beta, b0);
        end
    end
end

% Synthetic data
clear;
n = 200; p = 2;
X = [randn(n/2,p) + 2; randn(n/2,p) - 2];
y = [ones(n/2,1); -ones(n/2,1)];
[beta, b0, fhist] = svm_subgrad(X, y, 1, 5e+2, 1e-3);
title(sprintf("Synthetic: Binary SVM, Subgradient Method (t0 = %.2e)", t0));
plot(fhist(1,:) + eps, 'color', 'b'); xlabel('k'); ylabel('f'); hold on;
plot(fhist(2,:) + eps, 'color', 'm');
plot(fhist(3,:) + eps, 'color', 'k');
plot(fhist(4,:) + eps, 'color', 'c');
plot(fhist(5,:) + eps, 'color', 'g');
plot(fhist(6,:) + eps, 'color', [1 0.5 0]);
plot(fhist(7,:) + eps, 'color', [0 0.5 1]);
plot(cummin(fhist(8,:)) + eps, 'color', 'r'); hold off;
legend("5e-1", "1", "k^{1/5}", "k^{1/4}", "k^{1/3}", "k^{1/2}", "k", "Polyak");
hold off;

% w3a data
clear;
[y, X] = load_libsvm('w3a', 300);  % 300 = number of features in w3a
[beta, b0, fhist] = svm_subgrad(X, y, 1, 5e+2, 1e-3);
title(sprintf("w3a: Binary SVM, Subgradient Method (t0 = %.2e)", t0));
plot(fhist(1,:) + eps, 'color', 'b'); xlabel('k'); ylabel('f'); hold on;
plot(fhist(2,:) + eps, 'color', 'm');
plot(fhist(3,:) + eps, 'color', 'k');
plot(fhist(4,:) + eps, 'color', 'c');
plot(fhist(5,:) + eps, 'color', 'g');
plot(fhist(6,:) + eps, 'color', [1 0.5 0]);
plot(fhist(7,:) + eps, 'color', [0 0.5 1]);
plot(cummin(fhist(8,:)) + eps, 'color', 'r'); hold off;
legend("5e-1", "1", "k^{1/5}", "k^{1/4}", "k^{1/3}", "k^{1/2}", "k", "Polyak");
hold off;

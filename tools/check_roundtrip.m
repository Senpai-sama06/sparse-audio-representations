% Optional independent DCT/window cross-check. Requires no training pipeline.
rng(0);
N = 512; H = 128; T = 16000;
x = randn(T, 1);
n = (0:N-1)'; k = 0:N-1;
D = sqrt(2/N) * cos(pi/N * (n + .5) * k);
D(:, 1) = D(:, 1) / sqrt(2);
w = sqrt(.5 - .5 * cos(2*pi*n/N));
left = N-H;
right = left + mod(-(T+2*left-N), H);
xp = [zeros(left, 1); x; zeros(right, 1)];
y = zeros(size(xp)); weights = zeros(size(xp));
for start = 1:H:(length(xp)-N+1)
    idx = start:start+N-1;
    a = D' * (xp(idx) .* w);
    y(idx) = y(idx) + (D*a) .* w;
    weights(idx) = weights(idx) + w.^2;
end
y = y ./ max(weights, eps);
y = y(left+1:left+T);
relative_error = norm(y-x)/norm(x);
fprintf('Relative reconstruction error: %.3g\n', relative_error);
assert(relative_error < 1e-10);

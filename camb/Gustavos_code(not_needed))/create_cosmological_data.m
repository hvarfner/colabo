
num_points = 1000000;
outputfile = 'grid_cosmological';
save_iter  = 100;

seed = 0;
dim  = 9;
xmin = [-0.1, 0.7, 0.100, 0.02, 0.95, 0.65, -0.02, 1.0, 30];
xmax = [ 0.3, 0.8, 0.105, 0.3 , 0.96, 0.7,   0.01, 2.0, 10];

rng(seed);

map_to_problem  = @(x) bsxfun(@plus, bsxfun(@times, x, (xmax-xmin)), xmin);
f               = @(x) lrgLogLiklWrap(map_to_problem(x)); 

p = sobolset(dim);
p = scramble(p,'MatousekAffineOwen');
X = net(p,num_points);

data = zeros(num_points, dim + 2);
for i=1:num_points, 
  tic;
  try 
	y = f(X(i,:));
  catch
	y = NaN;
  end
  t=toc;
  row = [X(i,:), y, t];
  data(i, :) = row;
  disp([i, X(1,1), y, t])
  if mod(i,save_iter) == 0
   fprintf('Saved.\n\n') 
   save(outputfile, 'data') 
  end
end
save(outputfile, 'data')

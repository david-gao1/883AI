function pq = pq_create_matlab(N)
% PQ_CREATE_MATLAB - Pure MATLAB implementation of priority queue
% Creates a max-heap priority queue structure
% 
% pq = pq_create_matlab(N)
% N: maximum number of elements

pq.heap = [];  % [idx, cost] pairs
pq.size = 0;
pq.max_size = N;
pq.idx_map = containers.Map('KeyType', 'double', 'ValueType', 'double'); % maps idx to heap position

end


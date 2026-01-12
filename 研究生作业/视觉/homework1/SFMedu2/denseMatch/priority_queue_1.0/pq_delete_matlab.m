function pq = pq_delete_matlab(pq)
% PQ_DELETE_MATLAB - Pure MATLAB implementation of pq_delete
% Clears the priority queue (MATLAB handles memory automatically)
pq.heap = [];
pq.size = 0;
pq.idx_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
end


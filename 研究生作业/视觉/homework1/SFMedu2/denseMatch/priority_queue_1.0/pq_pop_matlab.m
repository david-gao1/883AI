function [pq, idx, cost] = pq_pop_matlab(pq)
% PQ_POP_MATLAB - Pure MATLAB implementation of pq_pop
% Removes and returns the topmost element

if pq.size == 0
    idx = [];
    cost = [];
    return;
end

idx = pq.heap(1, 1);
cost = pq.heap(1, 2);

% Remove from index map
pq.idx_map.remove(idx);

% Move last element to root
if pq.size > 1
    pq.heap(1, :) = pq.heap(pq.size, :);
    pq.idx_map(pq.heap(1, 1)) = 1;
    pq.size = pq.size - 1;
    pq = bubble_down_pq(pq, 1);
else
    pq.size = 0;
end

end

function pq = bubble_down_pq(pq, pos)
% Bubble down element at position pos
while true
    left = 2 * pos;
    right = 2 * pos + 1;
    largest = pos;
    
    if left <= pq.size && pq.heap(left, 2) > pq.heap(largest, 2)
        largest = left;
    end
    if right <= pq.size && pq.heap(right, 2) > pq.heap(largest, 2)
        largest = right;
    end
    
    if largest == pos
        break;
    end
    
    % Swap
    temp = pq.heap(pos, :);
    pq.heap(pos, :) = pq.heap(largest, :);
    pq.heap(largest, :) = temp;
    
    % Update index map
    pq.idx_map(pq.heap(pos, 1)) = pos;
    pq.idx_map(pq.heap(largest, 1)) = largest;
    
    pos = largest;
end
end


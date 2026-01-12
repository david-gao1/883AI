function pq = pq_push_matlab(pq, idx, cost)
% PQ_PUSH_MATLAB - Pure MATLAB implementation of pq_push
% Inserts or updates an element in the priority queue

if pq.idx_map.isKey(idx)
    % Update existing element
    pos = pq.idx_map(idx);
    old_cost = pq.heap(pos, 2);
    pq.heap(pos, 2) = cost;
    
    % If cost increased, bubble up; if decreased, bubble down
    if cost > old_cost
        pq = bubble_up(pq, pos);
    else
        pq = bubble_down(pq, pos);
    end
else
    % Insert new element
    pq.size = pq.size + 1;
    if pq.size > size(pq.heap, 1)
        pq.heap = [pq.heap; zeros(1000, 2)]; % grow array
    end
    pq.heap(pq.size, :) = [idx, cost];
    pq.idx_map(idx) = pq.size;
    pq = bubble_up(pq, pq.size);
end

end

function pq = bubble_up(pq, pos)
% Bubble up element at position pos
while pos > 1
    parent = floor(pos / 2);
    if pq.heap(parent, 2) >= pq.heap(pos, 2)
        break;
    end
    % Swap
    temp = pq.heap(parent, :);
    pq.heap(parent, :) = pq.heap(pos, :);
    pq.heap(pos, :) = temp;
    
    % Update index map
    pq.idx_map(pq.heap(parent, 1)) = parent;
    pq.idx_map(pq.heap(pos, 1)) = pos;
    
    pos = parent;
end
end

function pq = bubble_down(pq, pos)
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


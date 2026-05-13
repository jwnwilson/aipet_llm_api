import heapq

def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    heap = []  # end times of active meetings
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)       # reuse the room
        heapq.heappush(heap, end)
    return len(heap)

print(min_meeting_rooms([[0, 30], [5, 10], [15, 20]]))  # Output: 2
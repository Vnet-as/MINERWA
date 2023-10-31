"""
Class for flow interval tracking.

Keeps information about logged flows and allows querying for concurrent flows and other statistics
such as inter-flow arrival times.
"""

import statistics


class FlowList:
    """Due to optimization purposes, logged flows are not sorted when adding, but a sort() method
    needs to be called when querying the flow information."""

    def __init__(self) -> None:
        """Initializes empty flowlist."""
        self.data = []      # Flow storage list


    def __len__(self) -> int:
        """Returns the length of the FlowList."""
        return len(self.data)


    def empty(self) -> None:
        """Empties the flowlist, reseting it to its default state."""
        self.data = []


    def add(self, flow_start, flow_end) -> None:
        """Updates the flowlist by appending a new flow to the internal structures.

        Parameters:
            flow_start -- Start timestamp of the flow to be added
            flow_end   -- End timestamp of the flow to be added"""

        self.data.append((flow_start, flow_end))


    def query_concurrent(self) -> int:
        """Returns the number of concurrent flows"""
        starts = [(x[0], 1) for x in self.data]     # Interval starts with signs
        ends   = [(x[1], -1) for x in self.data]    # Interval ends with signs
        int_signs = sorted(starts + ends)           # Interval starts + end merged with signs
        max_concurrent = 0                          # Maximum number of concurrent flows
        cur_concurrent = 0                          # Current number of concurrent flows

        for _, sign in int_signs:
            cur_concurrent += sign
            max_concurrent  = max(cur_concurrent, max_concurrent)

        return max_concurrent

    def query_inter_times(self, max_length) -> tuple:
        """Retrieves inter-flow times between flow starts.

        Parameters:
            max_length -- Maximum length of the flow to take into account

        Returns:
            tuple -- (avg, std, min, max), where:
                iflows_avg - Average time between inter-flow times
                iflows_std - Standard deviation of times between inter-flow time
                iflows_min - Minimum inter-flow time logged
                iflows_max - Maximum inter-flow time logged"""

        iflows_avg = 0              # Interflos times mean
        iflows_std = 0              # Interflows times standard deviations
        iflows_min = 0              # Interflows times min
        iflows_max = 0              # Interflows times max
        last_flow_start = None      # Ending timestamp of the last flow
        interflow_times = []        # Computed inter-flow times

        # Select only rows that do not exceed the max length
        flows = [(f_start, f_end) for f_start, f_end in self.data if f_end - f_start <= max_length]

        # Sort the flows by its flow start
        flows.sort()

        # Compute the inter-flow times
        for f_start, _ in flows:
            if last_flow_start is None:
                last_flow_start = f_start
            else:
                interflow_times.append(int(f_start - last_flow_start))
                last_flow_start = f_start

        # Compute inter-flow statistics if all the flows have not been removed
        if interflow_times:
            iflows_avg = statistics.mean(interflow_times)
            iflows_std = statistics.stdev(interflow_times) if len(interflow_times) > 1 else 0
            iflows_min = min(interflow_times)
            iflows_max = max(interflow_times)

        return iflows_avg, iflows_std, iflows_min, iflows_max

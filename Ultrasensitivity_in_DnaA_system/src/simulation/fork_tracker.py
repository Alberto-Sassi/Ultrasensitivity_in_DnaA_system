"""
fork_tracker.py

Keeps the record of the origins that have fired and when they have fired. 
Determines when an origin fires on the basis of the firing probability rate. 

"""

import random
import numpy as np

class Origin:
    """
        Represents a replication origin.

        Attributes:
            origin_id (int): Unique identifier for the origin.
            parent_origin_id (int): ID of the parent origin (if any).
            created_at (float): Time at which the origin was created.
            firing_time (float or None): Time of the most recent firing event, or None if it hasn't fired yet.
    """
    def __init__(self, origin_id, parent_origin_id, created_at):
        self.origin_id = origin_id
        self.parent_origin_id = parent_origin_id
        self.created_at = created_at
        self.firing_time = None

    def eligible_to_fire(self, current_time, cfg):
        if self.firing_time is None:
            return True
        return (current_time - self.firing_time) > cfg.model.ECLIPSE
    
class TreeManager:
    """
        Manages the replication dynamics, including the firing of specific origins and termination 
        and division events.
    """

    def __init__(self, cfg):
        """
            Attributes:
                n_forks (int): Number of active replication forks.
                firing_probability_rate (float): Rate at which origins attempt to initiate replication.
                origins (dict): Maps origin IDs to `Origin` instances.
                current_time (float): Current simulation time.
                next_origin_id (int): ID to assign to the next newly created origin.
                division_scheduled (list): Times at which cell division is scheduled.
                termination_scheduled (list): Times of scheduled replication termination events.
                initiation_scheduled (list of lists): Scheduled initiation times and corresponding origin IDs.
                volume (float): Current cell volume.
                dt (float): Simulation time step.
                n_tot (float): Total number of DnaA binding sites in the cell.
                licensing (float): Delay between origin firing and actual initiation.
                multifork (list): List of active forks, together with the time they were created and the 
                    corresponding origins.
        """
        self.n_forks=0
        self.firing_probability_rate = 0.02
        self.origins = {}
        self.current_time = 0
        self.next_origin_id = 1
        self.division_scheduled = [] 
        self.termination_scheduled = []
        self.initiation_scheduled = [[],[]]
        self.volume = 1.
        self.dt = 0.01
        self.n_tot = cfg.model.SITES
        self.licensing = cfg.model.LICENSING 
        self.multifork = []

    def add_initial_origin(self):
        """
        Adds the first origin to the system at the current simulation time.

        This origin has no parent and is considered the root of the replication tree.
        It is assigned a unique ID and stored in the `origins` dictionary.
        """
        initial_origin = Origin(origin_id=self.next_origin_id, parent_origin_id=None, created_at=self.current_time)
        self.origins[initial_origin.origin_id] = initial_origin
        self.next_origin_id += 1

    def get_eligible_origins(self, cfg):
        """
        Returns a list of origin IDs that are eligible to fire at the current simulation time.

        An origin is considered eligible if:
        - It passes its internal firing criteria (`eligible_to_fire`).
        - It is not already scheduled for initiation.

        Returns:
            List[int]: IDs of origins ready to initiate replication.
        """
        firing_origins = []
        for origin in self.origins.values():
            if origin.eligible_to_fire(self.current_time, cfg) and origin.origin_id not in self.initiation_scheduled[1]:
                firing_origins.append(origin.origin_id)
        return firing_origins

    def schedule_initiation(self, origin_id, cfg):
        """
            Schedules a replication initiation event for the specified origin.

            The initiation is set to occur after a fixed licensing delay (cfg.model.LICENSING)
            from the current simulation time, and the origin ID is recorded for future processing.
        """
        self.initiation_scheduled[0].append(self.current_time+cfg.model.LICENSING)
        self.initiation_scheduled[1].append(origin_id)

    def process_eligible_origins(self, cfg):
        """
            Each eligible origin can fire with a probability rate 'firing_probability_rate'.
            If the origin fires, it is scheduled for initiation after the licensing delay.
        """
        firing_origins = self.get_eligible_origins(cfg)
        for origin_id in firing_origins:
            if random.random() < 1. - np.exp(-self.firing_probability_rate * self.dt):
                self.schedule_initiation(origin_id, cfg)

    def perform_initiations(self, cfg):
        """
            Performs the initiation of scheduled origins.
        """
        while self.initiation_scheduled[0] and self.initiation_scheduled[0][0] < self.current_time:
            if self.initiation_scheduled[1][0] in self.origins:
                self.fire_origin(self.initiation_scheduled[1][0], cfg)
            self.initiation_scheduled[0].pop(0)
            self.initiation_scheduled[1].pop(0)

    def fire_origin(self, origin_id, cfg):
        """
            Fires the specified origin and creates a new child origin.

            This method:
            - Sets the firing time of the given origin.
            - Creates a new origin with the current origin as its parent.
            - Adds the new origin to the system.
            - Increments the number of active replication forks.
            - Adds the firing event to the multifork list for future site availability calculations.
            - Schedules termination of the replication event.
        """
        origin = self.origins[origin_id]
        origin.firing_time = self.current_time
        new_origin = Origin(origin_id=self.next_origin_id, parent_origin_id=origin_id, created_at=self.current_time)
        new_origin.firing_time = self.current_time
        self.origins[new_origin.origin_id] = new_origin
        self.next_origin_id += 1
        self.n_forks += 2
        self.multifork.append([(origin_id, new_origin.origin_id), self.current_time])
        self.termination_scheduled.append((self.current_time + cfg.model.REP_TIME, (origin_id, new_origin.origin_id) ))

    def perform_termination(self, cfg):
        """
            Checks whether any replication termination events are scheduled at the current time.
            If so, performs them by:
                - Reducing the number of active replication forks.
                - Removing the parent-child link between the two origins involved in the terminating replication,
                so that they become independent ancestor origins of their respective chromosomes.
                - Removing the corresponding multifork entry.
                - Scheduling cell division after a delay (D), if termination triggers division.

            This ensures that once replication ends, the origins are correctly reclassified for future replication cycles,
            and the cell division machinery is activated at the appropriate time.
        """
        divide=False
        while self.termination_scheduled and self.current_time >= self.termination_scheduled[0][0]:
            if self.termination_scheduled[0][1][0] in self.origins.keys():           
                divide = self.current_time + cfg.model.D
                self.n_forks -= 2
                daughter_id=self.termination_scheduled[0][1][1]
                daughter=self.origins[daughter_id]
                daughter.parent_origin_id=None
                self.multifork.pop(0)

            self.termination_scheduled.pop(0)
        if divide:
            self.schedule_division(divide)

    def schedule_division(self, division_time):
        self.division_scheduled.append(division_time)

    def perform_division(self, cfg):
        """
            Performs cell division if the scheduled division time has been reached.

            The function:
                - Identifies ancestor origins (origins without a parent), each representing the root of a distinct genome.
                - Selects half of these ancestors to represent the genomes inherited by the daughter cell.
                - Reconstructs the full replication tree for each selected ancestor.
                - Retains only the origins and forks associated with these selected trees, discarding others.
                - Halves the cell volume.
                - Recalculates the total number of DnaA binding sites (n_tot) based on the updated replication profile.
                - Updates the number of active replication forks.
                - Removes the processed division event from the schedule.

            This allows the simulation to support multifork replication and division into cells with multiple chromosomes,
            while maintaining biological consistency in origin inheritance.
        """
        if self.division_scheduled and self.current_time >= self.division_scheduled[0]:
            ancestors = [ancestor for ancestor in self.origins.values() if ancestor.parent_origin_id==None]
            genomes=np.max((1, int(len(ancestors)/2)))
            selected_ancestors = random.sample(ancestors, genomes)
            selected_trees = []
            available_origins = []
            for ancestor in selected_ancestors:
                a_tree = self.get_tree(ancestor.origin_id)
                selected_trees.append(a_tree)
                available_origins += a_tree

            self.origins = {origin_id: self.origins[origin_id] for origin_id in available_origins}
            self.multifork = [fork for fork in self.multifork if fork[0][0] in available_origins]            
            self.volume /= 2
            self.n_tot=cfg.model.SITES*(genomes+np.sum([self.current_time-fork[1] for fork in self.multifork])/cfg.model.REP_TIME)
            self.n_forks = len(self.multifork)*2.
            self.division_scheduled.pop(0)

    def get_tree(self, root_id):
        """
            Returns the full replication tree rooted at the specified origin.

            Starting from the given root origin ID, this function recursively traverses all descendant origins
            (i.e., those whose parent_origin_id matches the current origin), collecting their IDs into a list.

            The resulting list represents the full lineage of replication events stemming from the root,
            and is used to reconstruct genome structure during division or analysis.
        """
        tree = []
        def traverse(origin_id):
            tree.append(origin_id)
            for child_id in [o.origin_id for o in self.origins.values() if o.parent_origin_id == origin_id]:
                traverse(child_id)
        traverse(root_id)
        return tree

    def update(self, time, fpr, volume, n_tot):
        '''
            this update is performed at every simulation step. 
        '''
        self.current_time=time
        self.firing_probability_rate = fpr
        self.volume = volume
        self.n_tot = n_tot

    def update_time(self):
        self.current_time += self.dt

    def simulate_step(self, cfg, update_time=False):
        '''
            Simulates a single step in the cell cycle model.

            This function:
                - Schedules future initiations for licensed origins ("fires" origins)
                - Executes scheduled initiations
                - Executes scheduled terminations
                - Executes scheduled cell divisions

            Optionally updates the simulation time (mainly for debugging purposes).
            In the simulations relevant to the paper, time updates are handled outside of tree_manager.
        '''
        if update_time:
            self.update_time()
        self.process_eligible_origins(cfg)
        self.perform_initiations(cfg) 
        self.perform_termination(cfg)
        self.perform_division(cfg) 

    def visualize_tree(self):
        children_map = {}
        for origin in self.origins.values():
            parent_id = origin.parent_origin_id
            if parent_id is not None:
                children_map.setdefault(parent_id, []).append(origin.origin_id)

        def build_subtree(origin_id):
            return {
                origin_id: [build_subtree(child_id) for child_id in children_map.get(origin_id, [])]
            }

        roots = [o.origin_id for o in self.origins.values() if o.parent_origin_id is None]
        return [build_subtree(root_id) for root_id in roots]

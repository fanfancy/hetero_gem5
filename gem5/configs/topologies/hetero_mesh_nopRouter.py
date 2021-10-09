# Copyright (c) 2010 Advanced Micro Devices, Inc.
#               2016 Georgia Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
from __future__ import absolute_import

from m5.params import *
from m5.objects import *

from common import FileSystemConfig

from topologies.BaseTopology import SimpleTopology

# Creates a generic Mesh assuming an equal number of cache
# and directory controllers.
# XY routing is enforced (using link weights)
# to guarantee deadlock freedom.

noc_link_width = 16 # todo modify this to parameters
nop_link_width = 8
noc_w = 5
nop_w = 3
noc_node_num = 20
nop_size = 6
core_num = noc_node_num*nop_size
router_num = core_num + nop_size


class hetero_mesh_nopRouter(SimpleTopology):
    description='hetero_mesh_nopRouter'

    def __init__(self, controllers):
        self.nodes = controllers

    # Makes a generic mesh
    # assuming an equal number of cache and directory cntrls

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        nodes = self.nodes

        num_routers = router_num

        # default values for link latency and router latency.
        # Can be over-ridden on a per link/router basis
        link_latency = options.link_latency # used by simple and garnet
        router_latency = options.router_latency # only used by garnet


        # There must be an evenly divisible number of cntrls to routers
        # Also, obviously the number or rows must be <= the number of routers
        cntrls_per_router, remainder = divmod(len(nodes), num_routers)

        # Create the noc routers
        routers = [Router(router_id=i, latency = router_latency,width = noc_link_width) \
            for i in range(core_num)]

        # Create the nop routers
        for i in range (nop_size):
            routers.append (Router(router_id=i+core_num, latency = router_latency,width = nop_link_width))
       
        network.routers = routers

        # link counter to set unique link ids
        link_count = 0

        # Add all but the remainder nodes to the list of nodes to be uniformly
        # distributed across the network.
        network_nodes = []
        remainder_nodes = []
        for node_index in range(len(nodes)):
            if node_index < (len(nodes) - remainder):
                network_nodes.append(nodes[node_index])
            else:
                remainder_nodes.append(nodes[node_index])

        print ("network_nodes",len(network_nodes))
        print ("remainder_nodes",len(remainder_nodes))

        # Connect each node to the appropriate router
        ext_links = []
        for (i, n) in enumerate(network_nodes):
            cntrl_level, router_id = divmod(i, num_routers)
            assert(cntrl_level < cntrls_per_router)
            if router_id >= core_num: ext_width = nop_link_width
            else: ext_width = noc_link_width
            ext_links.append(ExtLink(link_id=link_count, ext_node=n,
                                    width = ext_width,
                                    int_node=routers[router_id],
                                    latency = link_latency))
            link_count += 1

        # Connect the remainding nodes to router 0.  These should only be
        # DMA nodes.
        for (i, node) in enumerate(remainder_nodes):
            assert(node.type == 'DMA_Controller')
            assert(i < remainder)
            ext_links.append(ExtLink(link_id=link_count, ext_node=node,
                                    width = noc_link_width,
                                    int_node=routers[0],
                                    latency = link_latency))
            link_count += 1

        network.ext_links = ext_links

        # --------------- Create the noc links. ---------------
        
        int_links = []
        noc_num_rows= int(noc_node_num / noc_w)
        noc_num_columns = noc_w
        
        for chip_id in range (nop_size):
            id_offset = chip_id*noc_node_num
            # East output to West input links (weight = 1)
            for row in range(noc_num_rows):
                for col in range(noc_num_columns):
                    if (col + 1 < noc_num_columns):
                        east_out = col + (row * noc_num_columns) + id_offset
                        west_in = (col + 1) + (row * noc_num_columns) + id_offset
                        int_links.append(IntLink(link_id=link_count,
                                                src_node=routers[east_out],
                                                dst_node=routers[west_in],
                                                src_outport="East",
                                                dst_inport="West",
                                                width = noc_link_width,
                                                latency = link_latency,
                                                weight=1))
                        link_count += 1

            # West output to East input links (weight = 1)
            for row in range(noc_num_rows):
                for col in range(noc_num_columns):
                    if (col + 1 < noc_num_columns):
                        east_in = col + (row * noc_num_columns) + id_offset
                        west_out = (col + 1) + (row * noc_num_columns) + id_offset
                        int_links.append(IntLink(link_id=link_count,
                                                src_node=routers[west_out],
                                                dst_node=routers[east_in],
                                                src_outport="West",
                                                dst_inport="East",
                                                width = noc_link_width,
                                                latency = link_latency,
                                                weight=1))
                        link_count += 1

            # North output to South input links (weight = 2)
            for col in range(noc_num_columns):
                for row in range(noc_num_rows):
                    if (row + 1 < noc_num_rows):
                        north_out = col + (row * noc_num_columns) + id_offset
                        south_in = col + ((row + 1) * noc_num_columns) + id_offset
                        int_links.append(IntLink(link_id=link_count,
                                                src_node=routers[north_out],
                                                dst_node=routers[south_in],
                                                src_outport="North",
                                                dst_inport="South",
                                                width = noc_link_width,
                                                latency = link_latency,
                                                weight=2))
                        link_count += 1

            # South output to North input links (weight = 2)
            for col in range(noc_num_columns):
                for row in range(noc_num_rows):
                    if (row + 1 < noc_num_rows):
                        north_in = col + (row * noc_num_columns) + id_offset
                        south_out = col + ((row + 1) * noc_num_columns) + id_offset
                        int_links.append(IntLink(link_id=link_count,
                                                src_node=routers[south_out],
                                                dst_node=routers[north_in],
                                                src_outport="South",
                                                dst_inport="North",
                                                width = noc_link_width,
                                                latency = link_latency,
                                                weight=2))
                        link_count += 1

        # --------------- Create the nop links. ---------------
        nop_num_rows= int(nop_size / nop_w) 
        nop_num_columns = nop_w
        id_offset = core_num 

        for row in range(nop_num_rows):
            for col in range(nop_num_columns):
                if (col + 1 < nop_num_columns):
                    east_out = col + (row * nop_num_columns) + id_offset
                    west_in = (col + 1) + (row * nop_num_columns) + id_offset
                    int_links.append(IntLink(link_id=link_count,
                                            src_node=routers[east_out],
                                            dst_node=routers[west_in],
                                            src_outport="East",
                                            dst_inport="West",
                                            width = nop_link_width,
                                            latency = link_latency,
                                            weight=1))
                    link_count += 1

        # West output to East input links (weight = 1)
        for row in range(nop_num_rows):
            for col in range(nop_num_columns):
                if (col + 1 < nop_num_columns):
                    east_in = col + (row * nop_num_columns) + id_offset
                    west_out = (col + 1) + (row * nop_num_columns) + id_offset
                    int_links.append(IntLink(link_id=link_count,
                                            src_node=routers[west_out],
                                            dst_node=routers[east_in],
                                            src_outport="West",
                                            dst_inport="East",
                                            width = nop_link_width,
                                            latency = link_latency,
                                            weight=1))
                    link_count += 1

        # North output to South input links (weight = 2)
        for col in range(nop_num_columns):
            for row in range(nop_num_rows):
                if (row + 1 < nop_num_rows):
                    north_out = col + (row * nop_num_columns) + id_offset
                    south_in = col + ((row + 1) * nop_num_columns) + id_offset
                    int_links.append(IntLink(link_id=link_count,
                                            src_node=routers[north_out],
                                            dst_node=routers[south_in],
                                            src_outport="North",
                                            dst_inport="South",
                                            width = nop_link_width,
                                            latency = link_latency,
                                            weight=2))
                    link_count += 1

        # South output to North input links (weight = 2)
        for col in range(nop_num_columns):
            for row in range(nop_num_rows):
                if (row + 1 < nop_num_rows):
                    north_in = col + (row * nop_num_columns) + id_offset
                    south_out = col + ((row + 1) * nop_num_columns) + id_offset
                    int_links.append(IntLink(link_id=link_count,
                                            src_node=routers[south_out],
                                            dst_node=routers[north_in],
                                            src_outport="South",
                                            dst_inport="North",
                                            width = nop_link_width,
                                            latency = link_latency,
                                            weight=2))
                    link_count += 1


        for i in range (nop_size):
            nop_router_id = i + core_num
            noc_router_id = i * noc_node_num

            int_links.append(IntLink(link_id=link_count,
                width =  nop_link_width,
                src_node=routers[noc_router_id],
                dst_node=routers[nop_router_id],
                src_outport="East",
                dst_inport="West",
                src_serdes = True,
                latency = link_latency,
                weight=1))
            link_count += 1

            int_links.append(IntLink(link_id=link_count,
                width =  nop_link_width,
                src_node=routers[nop_router_id],
                dst_node=routers[noc_router_id],
                src_outport="West",
                dst_inport="East",
                dst_serdes = True,
                latency = link_latency,
                weight=1))
            link_count += 1

        network.int_links = int_links

    # Register nodes with filesystem
    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node([i],
                    MemorySize(options.mem_size) / options.num_cpus, i)

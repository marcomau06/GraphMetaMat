import networkx as nx
import numpy as np
import torch
from collections import defaultdict
from copy import deepcopy

from src.utils import plot_3d

plane_vector1 = np.array([0,-1,1]) # g1 -> g2
plane_vector2 = np.array([-1,0,1]) # g2 -> g3 <TP = U(g1, g2, g3)>
plane_vector3 = np.array([-1,1,0]) # TP -> g4 <cube = U(TP, g4)>
plane_vector4 = np.array([-1,0,0]) # cube -> g5 <RP = U(...)>
plane_vector5 = np.array([0,-1,0]) # RP -> ... <U...>
plane_vector6 = np.array([0,0,-1]) # ... -> ... <U...>

def tesselate_torch(points, edge_index, graph_index, start='tetrahedron', end='unitcell', **kwargs):
    assert start in ['tetrahedron', 'octet']
    assert end in ['octet', 'unitcell']

    # plot_3d(g, f'{g.graph["gid"]}-rm-orig.png')
    if start == 'tetrahedron':
        if end == 'octet':
            points, edge_index, graph_index = \
                tetrahedron2octet_torch(points, edge_index, graph_index, **kwargs)
        elif end == 'unitcell':
            points, edge_index, graph_index = \
                tetrahedron2octet_torch(points, edge_index, graph_index, **kwargs)
            points, edge_index, graph_index = \
                octet2unitcell_torch(points, edge_index, graph_index, **kwargs)
        else:
            assert False
    elif start == 'octet':
        if end == 'unitcell':
            points, edge_index, graph_index = \
                octet2unitcell_torch(points, edge_index, graph_index, **kwargs)
        else:
            assert False
    else:
        assert False

    return points, edge_index, graph_index

def tesselate(g, start='tetrahedron', end='unitcell', rm_redundant=True):
    if start == end:
        return g
    assert start in ['tetrahedron', 'octet']
    assert end in ['octet', 'unitcell']
    g = g.copy(as_view=False)

    # plot_3d(g, f'{g.graph["gid"]}-rm-orig.png')
    if start == 'tetrahedron':
        if end == 'octet':
            g = tetrahedron2octet(g)
        elif end == 'unitcell':
            g = tetrahedron2octet(g)
            g = octet2unitcell(g)
        else:
            assert False
    elif start == 'octet':
        if end == 'unitcell':
            g = octet2unitcell(g)
        else:
            assert False
    else:
        assert False

    if rm_redundant:
        g, _ = rm_redundant_nodes(g)

    # plot_3d(g, f'{g.graph["gid"]}-rm-pre.png')
    # plot_3d(g, f'{g.graph["gid"]}-rm-post.png')
    # for nid in g.nodes():
    #     # g.nodes[nid]['coord'] *= 5
    #     g.nodes[nid]['coord'] /= 2
    #     g.nodes[nid]['coord'] += 0.5

    return g

def untesselate(g, start='unitcell', end='tetrahedron', rm_redundant=True):
    if start == end:
        return g
    assert start in ['octet', 'unitcell']
    assert end in ['tetrahedron', 'octet']

    g = g.copy(as_view=False)

    # plot_3d(g, f'{g.graph["gid"]}-rm-orig.png')
    if start == 'unitcell':
        if end == 'octet':
            g = unitcell2octet(g)
        elif end == 'tetrahedron':
            g = unitcell2octet(g)
            g = octet2tetrahedron(g)
        else:
            assert False
    elif start == 'octet':
        if end == 'tetrahedron':
            g = octet2tetrahedron(g)
        else:
            assert False
    else:
        assert False

    if rm_redundant:
        g, _ = rm_redundant_nodes(g.copy())
    g.remove_edges_from(nx.selfloop_edges(g))
    return g

def tetrahedron2octet_torch(points, edge_index, graph_index, node_dict=None, edge_dict=None):
    points_new1, edge_index_new1, graph_index_new1, node_mask1, edge_mask1 = \
        reflect_points_wrapper_torch(
            points, edge_index, graph_index, plane_vector1, offset=points.shape[0])
    nid_index = torch.arange(points_new1.shape[0], device=points.device)
    nid_index[node_mask1] = torch.arange(torch.sum(node_mask1).item()) + points_new1.shape[0]
    points_new2, edge_index_new2, graph_index_new2, node_mask2, edge_mask2 = \
        reflect_points_wrapper_torch(
            points_new1, edge_index_new1, graph_index_new1, plane_vector2,
            nid_index=nid_index,
            offset=points.shape[0] + torch.sum(node_mask1))
    assert points.shape[0] + torch.sum(node_mask1) == nid_index.max() + 1
    points = \
        torch.cat((points, points_new1[node_mask1], points_new2[node_mask2]), dim=0)
    edge_index = \
        torch.cat((edge_index, edge_index_new1[:,edge_mask1], edge_index_new2[:,edge_mask2]), dim=-1)
    graph_index = \
        torch.cat((graph_index, graph_index_new1[node_mask1], graph_index_new2[node_mask2]), dim=0)
    if node_dict is not None and len(node_dict) > 0:
        for k in node_dict:
            node_dict[k] = \
                torch.cat((
                    node_dict[k], node_dict[k][node_mask1], node_dict[k][node_mask2]
                ), dim=0)
    if edge_dict is not None and len(edge_dict) > 0:
        for k in edge_dict:
            edge_dict[k] = \
                torch.cat((
                    edge_dict[k], edge_dict[k][edge_mask1], edge_dict[k][edge_mask2]
                ), dim=0)

    points_new, edge_index_new, graph_index_new, node_mask, edge_mask = \
        reflect_points_wrapper_torch(
            points, edge_index, graph_index, plane_vector3, offset=points.shape[0])
    points = torch.cat((points, points_new[node_mask]), dim=0)
    edge_index = torch.cat((edge_index, edge_index_new[:,edge_mask]), dim=-1)
    graph_index = torch.cat((graph_index, graph_index_new[node_mask]), dim=0)
    if node_dict is not None and len(node_dict) > 0:
        for k in node_dict:
            node_dict[k] = \
                torch.cat((node_dict[k], node_dict[k][node_mask]), dim=0)
    if edge_dict is not None and len(edge_dict) > 0:
        for k in edge_dict:
            edge_dict[k] = \
                torch.cat((edge_dict[k], edge_dict[k][edge_mask]), dim=0)

    assert points.shape[0] == graph_index.shape[0]
    assert edge_index.max() == points.shape[0]-1
    return points, edge_index, graph_index

def octet2unitcell_torch(points, edge_index, graph_index, node_dict=None, edge_dict=None):
    for plane_vector in [plane_vector4, plane_vector5, plane_vector6]:
        points_new, edge_index_new, graph_index_new, node_mask, edge_mask = \
            reflect_points_wrapper_torch(
                points, edge_index, graph_index, plane_vector, offset=points.shape[0])
        points = torch.cat((points, points_new[node_mask]), dim=0)
        edge_index = torch.cat((edge_index, edge_index_new[:,edge_mask]), dim=-1)
        graph_index = torch.cat((graph_index, graph_index_new[node_mask]), dim=0)
    if node_dict is not None and len(node_dict) > 0:
        for k in node_dict:
            node_dict[k] = \
                torch.cat((node_dict[k], node_dict[k][node_mask]), dim=0)
    if edge_dict is not None and len(edge_dict) > 0:
        for k in edge_dict:
            edge_dict[k] = \
                torch.cat((edge_dict[k], edge_dict[k][edge_mask]), dim=0)

    assert points.shape[0] == graph_index.shape[0]
    assert edge_index.max() == points.shape[0]-1
    return points, edge_index, graph_index

def tetrahedron2octet(g):
    # x'=x, y'=z, z'=y
    # x'=z, y'=y, z'=x
    # x'=y, y'=x, z'=z
    g2 = reflect_points_wrapper(g, plane_vector1)
    g3 = reflect_points_wrapper(g2, plane_vector2)
    g = nx.compose(g, g2)
    g = nx.compose(g, g3)
    g = nx.compose(g, reflect_points_wrapper(g, plane_vector3))
    return g

def octet2unitcell(g):
    # x'=x, y'=y, z'=-z
    # x'=x, y'=-y, z'=z
    # x'=-x, y'=y, z'=z
    for plane_vector in [plane_vector4, plane_vector5, plane_vector6]:
        g = nx.compose(g, reflect_points_wrapper(g, plane_vector))
    return g

def unitcell2octet(g):
    for plane_vector in [plane_vector6, plane_vector5, plane_vector4]:
        nid_li = list(g.nodes())
        node_class = \
            classify_points(
                np.array([g.nodes[nid]['coord'] for nid in nid_li]),
                plane_vector)

        cur_nid_new = max(nid_li) + 1
        for i, nid in enumerate(nid_li):
            if node_class[i] == 1:
                for nid_other in list(nx.neighbors(g, nid)):
                    if node_class[nid_li.index(nid_other)] == -1:
                        coord_new = line_segment_plane_intersection(
                            g.nodes[nid]['coord'],
                            g.nodes[nid_other]['coord'],
                            plane_vector
                        )
                        g.add_node(cur_nid_new, coord=coord_new)
                        g.add_edge(cur_nid_new, nid_other)
                        cur_nid_new += 1
                    # g.remove_edge(nid, nid_other)
                g.remove_node(nid)

        g = nx.convert_node_labels_to_integers(g)
    return g

def octet2tetrahedron(g):
    for plane_vector in [plane_vector3, plane_vector2, plane_vector1]:
        nid_li = list(g.nodes())
        node_class = \
            classify_points(
                np.array([g.nodes[nid]['coord'] for nid in nid_li]),
                plane_vector)

        cur_nid_new = max(nid_li) + 1
        for i, nid in enumerate(nid_li):
            if node_class[i] == 1:
                for nid_other in list(nx.neighbors(g, nid)):
                    if node_class[nid_li.index(nid_other)] == -1:
                        coord_new = line_segment_plane_intersection(
                            g.nodes[nid]['coord'],
                            g.nodes[nid_other]['coord'],
                            plane_vector
                        )
                        g.add_node(cur_nid_new, coord=coord_new)
                        g.add_edge(cur_nid_new, nid_other)
                        cur_nid_new += 1
                    # g.remove_edge(nid, nid_other)
                g.remove_node(nid)

        g = nx.convert_node_labels_to_integers(g)
    return g
def reflect_points_wrapper_torch(points, edge_index, graph_index, plane_vector, offset, nid_index=None, eps=1e-10):
    points_new = \
        reflect_points(points, plane_vector)
    graph_index_new = graph_index # NOTE: POINTER NOT VALUE!
    edge_index_new = torch.clone(edge_index)
    mask_node = torch.ge(torch.norm(points_new-points, dim=-1), eps)
    mask_edge = \
        torch.zeros(edge_index_new.shape[-1], dtype=torch.bool, device=mask_node.device)
    if True in mask_node:
        indices_to_update, _ = torch.sort(torch.nonzero(mask_node).view(-1))
        if nid_index is not None:
            assert nid_index.shape == (points_new.shape[0],)
            indices_to_update = nid_index[indices_to_update]
        for nid_new, nid_old in enumerate(indices_to_update):
            edge_index_mask_update = torch.eq(edge_index_new, nid_old)
            edge_index_new[edge_index_mask_update] = nid_new + offset
            mask_edge[
                torch.logical_or(edge_index_mask_update[0], edge_index_mask_update[1])
            ] = True
    return points_new, edge_index_new, graph_index_new, mask_node, mask_edge

import time
def reflect_points_wrapper(g, plane_vector, eps=1e-10):
    g2 = g.copy(as_view=False)
    g2_nid_li_ordered = sorted(g2.nodes())
    coord_li = [g2.nodes[nid]['coord'] for nid in g2_nid_li_ordered]
    coord_new_li = \
        reflect_points(np.array(coord_li), plane_vector)
    relabel_dict = {}
    for nid, coord_new in zip(g2_nid_li_ordered, coord_new_li):
        if np.linalg.norm(g2.nodes[nid]['coord'] - coord_new) < eps:
            pass
        else:
            g2.nodes[nid]['coord'] = coord_new
            relabel_dict[nid] = max(g.nodes()) + len(relabel_dict) + 1
    g2 = nx.relabel_nodes(g2, relabel_dict)
    return g2

# relabel_dict = {}
# for nid, coord_new in zip(g2_nid_li_ordered, coord_new_li):
#     if np.linalg.norm(g2.nodes[nid]['coord'] - coord_new) < eps:
#         pass
#     else:
#         g2.nodes[nid]['coord'] = coord_new
#         relabel_dict[nid] = max(g.nodes()) + len(relabel_dict) + 1
# g2 = nx.relabel_nodes(g2, relabel_dict)
# g = nx.compose(g, g2)
# return g, g2

def line_segment_plane_intersection(segment_start, segment_end, plane_vector):
    # Calculate the direction vector of the line segment
    plane_vector = plane_vector / np.linalg.norm(plane_vector)
    segment_direction = segment_end - segment_start

    # Calculate the parameter t where the line intersects the plane
    t = np.dot(plane_vector, -segment_start) / np.dot(plane_vector, segment_direction)

    # Check if the line segment is parallel to the plane
    if np.isinf(t) or np.isnan(t) or t < 0 or t > 1:
        assert False
        # return None  # No intersection, the line segment is parallel or outside the plane

    # Calculate the intersection point
    intersection_point = segment_start + t * segment_direction

    return intersection_point

def reflect_points_torch(points, plane_vector):
    # Normalize the plane vector
    plane_normal = (plane_vector / torch.norm(plane_vector)).reshape(1,3)

    # Calculate the projection of each point onto the plane
    projections = torch.matmul(points, plane_normal.transpose(0,1))[:, np.newaxis] * plane_normal

    # Calculate the reflection of each point across the plane
    reflections = points - 2 * projections

    return reflections

def reflect_points(points, plane_vector):
    # Normalize the plane vector
    plane_normal = plane_vector / np.linalg.norm(plane_vector)

    # Calculate the projection of each point onto the plane
    projections = np.dot(points, plane_normal)[:, np.newaxis] * plane_normal

    # Calculate the reflection of each point across the plane
    reflections = points - 2 * projections

    # # which nodes are on the tangent plane of the projection?
    # tangent_nodes = np.dot(points, plane_normal) < eps

    return reflections

def project_points(points, plane_vector):
    # Normalize the plane vector
    plane_normal = plane_vector / np.linalg.norm(plane_vector)

    # Calculate the projection of each point onto the plane
    projections = np.dot(points, plane_normal)[:, np.newaxis] * plane_normal

    # Calculate the reflection of each point across the plane
    reflections = points - projections

    return reflections

def classify_points(points, plane_vector, eps=1e-5):
    # Normalize the plane vector
    plane_normal = plane_vector / np.linalg.norm(plane_vector)

    # Calculate the signed distances from each point to the plane
    signed_distances = np.dot(points, plane_normal) # - np.dot(plane_normal, plane_normal) / np.linalg.norm(plane_normal)

    # Classify the points based on their signed distances
    sides = np.zeros_like(signed_distances)
    sides[signed_distances > eps] = +1
    sides[signed_distances < -eps] = -1

    return sides

def rm_double_edges(g, eps=1e-12):
    g_new = nx.Graph(**g.graph)

    # Mark all the vertices as not visited
    nid2ray_nid = defaultdict(dict)
    visited = [False] * g.number_of_nodes()

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    s, *_ = list(g.nodes())
    queue.append(s)
    visited[s] = True
    updated = False

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)

        # Get all adjacent vertices of the
        # dequeued vertex s.
        # If an adjacent has not been visited,
        # then mark it visited and enqueue it
        for adj in nx.neighbors(g, s):
            if not visited[adj]:
                queue.append(adj)
                visited[adj] = True

            ray = g.nodes[adj]['coord'] - g.nodes[s]['coord']
            magnitude = np.linalg.norm(ray)
            ray = ray/magnitude

            added_edge = False
            for two_hop in list(nid2ray_nid[s]):
                if adj == two_hop:
                    continue
                ray_two_hop, magnitude_two_hop = nid2ray_nid[s][two_hop]
                if np.dot(ray, ray_two_hop) > 1-eps:
                    if magnitude_two_hop == magnitude:
                        pass
                    elif magnitude_two_hop > magnitude:
                        # add 2 edges: delete the edge bw two_hop and s
                        nid2ray_nid[s].pop(two_hop)
                        nid2ray_nid[two_hop].pop(s)

                        g_new.add_edge(s, adj, **g.edges[s, adj])
                        assert s != adj
                        nid2ray_nid[s][adj] = ray, magnitude
                        nid2ray_nid[adj][s] = -ray, magnitude

                        g_new.add_edge(adj, two_hop, **g_new.edges[s, two_hop])
                        assert two_hop != adj
                        nid2ray_nid[adj][two_hop] = ray, magnitude_two_hop-magnitude
                        nid2ray_nid[two_hop][adj] = -ray, magnitude_two_hop-magnitude

                        g_new.remove_edge(s, two_hop)
                        updated = True
                    elif magnitude_two_hop < magnitude:
                        # add 1 edge
                        g_new.add_edge(two_hop, adj, **g.edges[s, adj])
                        # assert two_hop != adj
                        nid2ray_nid[two_hop][adj] = ray, magnitude-magnitude_two_hop
                        nid2ray_nid[adj][two_hop] = -ray, magnitude-magnitude_two_hop
                        updated = True
                    else:
                        assert False
                    added_edge = True
                    break

            if not added_edge:
                # add 1 edge
                g_new.add_edge(s, adj, **g.edges[s, adj])
                assert s != adj
                nid2ray_nid[s][adj] = ray, magnitude
                nid2ray_nid[adj][s] = -ray, magnitude


    assert all(visited)
    for nid in g_new.nodes():
        g_new.nodes[nid]['coord'] = g.nodes[nid]['coord']

    return g_new, updated


def rm_redundant_nodes(g):
    # TODO: make this faster
    updated = False
    while(True):
        nid_to_rm = None
        for nid in g.nodes():
            if len(set(nx.neighbors(g, nid)) - {nid}) == 2:
                nid_u, nid_v = list(set(nx.neighbors(g, nid)) - {nid})
                ray_u = g.nodes[nid_u]['coord']-g.nodes[nid]['coord']
                ray_v = g.nodes[nid]['coord']-g.nodes[nid_v]['coord']
                abscossim = \
                    np.abs(np.dot(ray_u, ray_v)/(np.linalg.norm(ray_u)*np.linalg.norm(ray_v)))
                if abscossim > 1-1e-12 and np.dot(ray_u, ray_v) > 0.0:
                    nid_to_rm = (nid, nid_u, nid_v)
        if nid_to_rm is None:
            break
        else:
            nid, nid_u, nid_v = nid_to_rm
            g.add_edge(nid_u, nid_v)
            g.edges[nid_u, nid_v].update(dict(g.edges[nid_u, nid]))
            g.remove_node(nid)
            updated = True

    g = nx.relabel_nodes(g, {nid:i for i,nid in enumerate(g.nodes())}, copy=True)
    return g, updated

if __name__ == '__main__':
    # g = nx.Graph()
    # g.add_node(0, coord=np.array([0.0,0.0,0.0]))
    # g.add_node(1, coord=np.array([0.5,0.0,0.0]))
    # g.add_node(2, coord=np.array([1.0,0.8,0.0]))
    # g.add_node(3, coord=np.array([1.0,1.0,1.0]))
    # g.add_edge(0, 1)
    # g.add_edge(1, 2)
    # g.add_edge(2, 3)

    g = nx.Graph()
    g.add_node(0, coord=np.array([0.5,0.0,0.0]))
    g.add_node(1, coord=np.array([0.5,0.5,0.0]))
    g.add_node(2, coord=np.array([0.5,0.5,0.5]))
    g.add_node(3, coord=np.array([0.0,0.0,0.0]))
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(0, 3)

    print([g.nodes()[x]['coord'] for x in g.nodes()])
    plot_3d(g, pn='/home/derek/Documents/MaterialSynthesis/pre-tesselate.png')
    g_t = tesselate(g)
    plot_3d(g_t, pn='/home/derek/Documents/MaterialSynthesis/post-tesselate.png')
    g = untesselate(g_t)
    print([g.nodes()[x]['coord'] for x in g.nodes()])
    plot_3d(g, pn='/home/derek/Documents/MaterialSynthesis/pre-tesselate2.png')

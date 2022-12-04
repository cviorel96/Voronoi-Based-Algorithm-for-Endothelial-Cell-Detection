import cv2 as cv
import os
import pathlib
import numpy as np
import math
from math import acos, degrees
import statistics as sts

np.set_printoptions(suppress=True)

'''
Endothelial cell detection algorithm with Voronoi Tesselation.

INSTRUCTIONS:
Press:
    L -> To show full image
    J - > To show nuclei
    (Any Key) -> To quit
'''

parent = pathlib.Path(__file__).parent.absolute()
file_path = os.path.join(parent, "Images")
os.chdir(file_path)


class Process:
    def __init__(self, path):
        self.img_list, self.resized, self.print_img = [], [], []
        self.path = path
        self.name = ''

    def add_image(self):
        for i in os.listdir(self.path):
            if not i.startswith('.'):
                self.img_list.append(i)
        self.img_list.sort()

    def load_image(self):
        print("What image would you like to open?")
        print(self.img_list)
        c_img = 'was1.tif'
        ret, img = cv.imreadmulti(self.img_list[self.img_list.index(c_img)], [], cv.IMREAD_ANYCOLOR)

        self.name = self.img_list[self.img_list.index(c_img)]
        self.resized = cv.resize(img[1], (962, 2037), interpolation=cv.INTER_CUBIC)
        self.print_img = cv.resize(img[0], (962, 2037), interpolation=cv.INTER_CUBIC)

    def open_image(self):
        """This function is disabled!"""
        cv.imshow(self.name, self.resized)
        cv.imshow(self.name, self.print_img)


class Analysis(Process):
    def __init__(self, path):
        super(Analysis, self).__init__(path)
        self.landmark_delaunay, self.area, self.circularity = [], [], []

    def contour_detection(self):
        b = cv.GaussianBlur(self.resized, (9, 9), cv.BORDER_DEFAULT)
        b = cv.cvtColor(b, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(b, 25, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY)

        contours, hier = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            hull = cv.convexHull(cnt)
            area_1 = cv.contourArea(hull)
            perimeter = cv.arcLength(hull, True)

            if perimeter > 0 and area_1 > 1000:
                circularity = 4 * math.pi * (area_1 / perimeter ** 2)
                hull2 = cv.convexHull(cnt, returnPoints=False)
                defects = cv.convexityDefects(cnt, hull2)
                defects = np.asarray(defects)
                np.where(defects is not None, defects, [[20, 20, 20, 20]])
                # mean_d = np.mean(defects, axis=0)
                # mean_d = mean_d.astype(int)
                # convex_d = mean_d[:, 3]
                a_max = np.amax(defects, axis=0)
                convex_d = a_max[:, 3]
                self.area.append(int(area_1))
                self.circularity.append(circularity)
            else:
                continue

            # 0.8; 4000
            if circularity > 0.1 and convex_d < 450000:
                m = cv.moments(cnt)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                self.landmark_delaunay.append((cx, cy))

                cv.drawContours(self.print_img, [cnt], -1, (0, 255, 255), 2, cv.LINE_AA)
                cv.drawContours(self.print_img, [hull], -1, (20, 134, 255), 3, cv.LINE_AA)
                cv.circle(self.print_img, (cx, cy), 8, (111, 165, 222), -1, cv.LINE_AA)

                cv.drawContours(self.resized, [cnt], -1, (0, 255, 255), 2, cv.LINE_AA)
                cv.drawContours(self.resized, [hull], -1, (20, 134, 255), 3, cv.LINE_AA)
                cv.circle(self.resized, (cx, cy), 8, (111, 165, 255), -1, cv.LINE_AA)
            else:
                continue


class Voronoi(Analysis):
    def __init__(self, path):
        super(Voronoi, self).__init__(path)
        self.facets, self.centers, self.distance, self.distance_mean, self.delaunay = [], [], [], [], []
        self.w_distance, self.adjacent_distance = [], []

    def get_facets(self):
        shape = self.resized.shape

        x, y, ch = shape
        rect = (0, 0, y, x)
        subdiv = cv.Subdiv2D(rect)
        subdiv.insert(self.landmark_delaunay)

        self.facets, self.centers = subdiv.getVoronoiFacetList([])

        self.centers = np.array(self.centers, dtype='int32')

        edge_list2 = subdiv.getEdgeList()
        edge_list = np.array(edge_list2, dtype='int32')

        return edge_list

    def distances(self, edge_list):
        for i, n in enumerate(self.centers):
            n = np.array(n)

            index = np.where(((edge_list[:, 0] == n[0]) & (edge_list[:, 1] == n[1])) |
                             ((edge_list[:, 2] == n[0]) & (edge_list[:, 3] == n[1])))
            index1 = np.unique(index[0])
            wrap = []
            w_wrap = []
            coordinates = []

            for t in edge_list[index1]:
                if any(t > 3000) or any(t < -3000):
                    continue
                elif np.array_equal(t[0:2], n):
                    xo, yo, xd, yd = t[0], t[1], t[2], t[3]
                else:
                    xo, yo, xd, yd = t[2], t[3], t[0], t[1]

                dif2 = ((xo - xd) ** 2 + (yo - yd) ** 2)
                dif1 = np.absolute(dif2)
                dif = np.sqrt(dif1)

                w_wrap.append([dif, dif])

                if dif < 1000:
                    coordinates.append([xo, yo, xd, yd])
                    wrap.append(dif)
                else:
                    continue

            # ----------------------------------
            # Remove distances that are outliers
            wrap = np.array(wrap, dtype='float')
            coordinates = np.array(coordinates, dtype='int32')
            mean = np.mean(wrap)
            std = np.std(wrap)
            dist_from_mean = abs(wrap - mean)

            max_devs = 1.4
            no_outlier = dist_from_mean < max_devs * std
            f_wrap = wrap[no_outlier]
            f_coordinates = coordinates[no_outlier]

            self.distance.append(f_wrap)
            self.w_distance.append(w_wrap)
            self.distance_mean.append([round(sts.mean(f_wrap)), i])
            self.adjacent_distance.append([[self.centers[i]], [round(sts.mean(f_wrap)), n[0]]])
            self.delaunay.append(f_coordinates)

        # --------------------------------------------
        # Prints the average distance of every nucleus
        for b in range(0, len(self.distance_mean)):
            tupl = tuple(self.centers[b])
            string = str(self.distance_mean[b])
            cv.putText(self.resized, string, tupl, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # -------------------------
        # Prints the Delaunay lines
        for d in self.delaunay:
            for c in d:
                cv.line(self.resized, tuple(c[0:2]), tuple(c[2:4]), (64, 51, 214), 3, cv.LINE_AA)
                cv.line(self.print_img, tuple(c[0:2]), tuple(c[2:4]), (64, 51, 214), 3, cv.LINE_AA)

    def tesselation(self):
        """This function is disabled!"""
        for i in range(0, len(self.facets)):
            facet_arr = np.empty((0, 2), dtype='int32')
            facet_arr = np.append(facet_arr, self.facets[i].astype(int), axis=0)

            # ----------------------------------
            # Prints the original Voronoi facets
            cv.polylines(self.resized, [facet_arr], True, (240, 240, 240), 3, cv.LINE_AA)
            cv.polylines(self.print_img, [facet_arr], True, (240, 240, 240), 3, cv.LINE_AA)

            # ----------------------------
            # Prints the original vertices
            for r in range(0, len(facet_arr)):
                cv.circle(self.resized, tuple(facet_arr[r]), 6, (0, 105, 217), -1, cv.LINE_AA)
                cv.circle(self.print_img, tuple(facet_arr[r]), 6, (0, 105, 217), -1, cv.LINE_AA)


class ReshapeVoronoi(Voronoi):
    def __init__(self, path):
        super(ReshapeVoronoi, self).__init__(path)
        self.nuclei, self.bound_nuclei, self.bound_nuclei_distance, self.bound_distance = [], [], [], []
        self.ortho_facets, self.ortho_nuclei, self.adjacent_average = [], [], []

    def find_orthogonal_facets(self):
        distance = np.asarray(self.distance_mean, dtype='int32')
        big_areas = []

        for i in range(0, len(self.facets)):
            delaunay_arr = np.empty((0, 4), dtype='int32')
            delaunay_arr = np.append(delaunay_arr, self.delaunay[i], axis=0)
            w_distance = np.empty((0, 2), dtype='float')
            w_distance = np.append(w_distance, self.w_distance[i])
            w_distance = np.reshape(w_distance, (-1, 2))

            # 530; 20000
            if distance[i, 0] > 100 and self.area[i] > 100:
                self.bound_nuclei.append(self.centers[i])
                self.nuclei.append(delaunay_arr[:, 2:])
                c_centers = self.centers[i]
                c_nuclei = delaunay_arr[:, 2:]
                self.bound_nuclei_distance.append(distance[i])

                if 20000 < self.area[i] < 25000:
                    big_areas.append([self.area[i] / 1, self.area[i] / 1])
                else:
                    big_areas.append([self.area[i], self.area[i]])

                self.bound_distance.append(w_distance)

                # -----------------------------------------------------
                # Calculates the slopes of the delaunay lines and gives
                # the angles between the lines and the x-axis
                big_angles = []
                for m in range(0, len(c_nuclei)):
                    ydd = c_centers[1] - c_nuclei[m, 1]
                    xdd = c_centers[0] - c_nuclei[m, 0]

                    ydd1 = np.delete(ydd, np.where(ydd == 0)[0])
                    xdd1 = np.delete(xdd, np.where(xdd == 0)[0])

                    m2 = ydd1 / xdd1
                    tan_b = math.atan(m2) * 180 / math.pi
                    big_angles.append(tan_b)

                good_nuclei = []
                b = []
                for n in range(0, len(c_nuclei)):
                    var1 = np.where(self.centers == delaunay_arr[n, 2:])[0]
                    var2 = self.facets[var1[0]]
                    var3 = np.asarray(var2, dtype='float')

                    # --------------------------------------------------
                    # Calculates the slopes of the facet lines and gives
                    # the angles between the lines and the x-axis
                    for m in range(0, len(var3)):
                        yd = var3[m, 1] - var3[:, 1]
                        xd = var3[m, 0] - var3[:, 0]

                        yd1 = np.delete(yd, np.where(yd == 0)[0][0])
                        xd1 = np.delete(xd, np.where(xd == 0)[0][0])
                        m1 = yd1 / xd1

                        for z in range(0, len(m1)):
                            tan_a = math.atan(m1[z]) * 180 / math.pi
                            g = tan_a - big_angles[n]

                            if 89.9 < g < 90.1:
                                b.append((var3[m, 0], var3[m, 1]))
                                good_nuclei.append(self.centers[var1[0]])
                            elif -89.9 > g > -90.1:
                                b.append((var3[m, 0], var3[m, 1]))
                                good_nuclei.append(self.centers[var1[0]])
                            else:
                                continue

                good_nuclei = np.array(good_nuclei)
                good_nuclei = np.reshape(good_nuclei, (-1, 2))
                _, idx = np.unique(good_nuclei, axis=0, return_index=True)
                if b:
                    self.ortho_facets.append(b)
                    self.ortho_nuclei.append(good_nuclei[np.sort(idx)])
                else:
                    continue

                # ------------------------------------
                # Prints the orthogonal Voronoi facets
                # for gg in self.ortho_facets:
                #     gg = np.asarray(gg, dtype='int32')
                #
                #     for w in range(0, len(gg), 2):
                #         cv.line(self.resized, tuple(gg[w]), tuple(gg[w + 1]), (0, 255, 0), 3, cv.LINE_AA)
            else:
                continue

        return big_areas

    def differences(self):
        ortho_facets = []

        for i in range(0, len(self.nuclei)):
            crt = []
            a = np.array(self.ortho_facets[i], dtype='int32')

            for k in range(2, len(a) + 2, 2):
                if (a[k - 2, 0] - a[k - 1, 0]) == 0 and (a[k - 2, 1] - a[k - 1, 1]) == 0:
                    continue
                else:
                    crt.append(a[k - 2])
                    crt.append(a[k - 1])
            crt = np.asarray(crt)

            ortho_facets.append(crt)

        return ortho_facets

    def barycentric_coordinates(self, ortho_facets, big_areas):
        mod2 = []
        dom = []
        dd2 = []

        for k in range(0, len(self.nuclei)):
            nuclei = np.empty((0, 0), dtype='int32')
            nuclei = np.append(nuclei, np.floor(self.nuclei[k]).astype(int))
            nuclei = np.reshape(nuclei, (-1, 2))

            for g in range(0, len(self.adjacent_distance)):
                d_1 = np.array(self.adjacent_distance[g][0][0][0])
                d_2 = np.array(self.adjacent_distance[g][0][0][1])
                d_3 = np.array(self.adjacent_distance[g][1])

                indx = np.where((nuclei[:, 0] == d_1) & (nuclei[:, 1] == d_2))
                indx = np.unique(indx[0])

                if indx.size > 0:
                    nuclei[indx] = d_3
                else:
                    continue

            self.adjacent_average.append(nuclei)

        # ---------------------------------------------------------------------
        # Takes the indices of common vertices and appends them to com_indices
        for i in range(0, len(ortho_facets)):
            com_indices = []
            for y in range(0, len(ortho_facets[i])):
                v_index = np.where((ortho_facets[i][y, 0] == ortho_facets[i][:, 0]) &
                                   (ortho_facets[i][y, 0] == ortho_facets[i][:, 0]))

                if v_index[0].size > 1:
                    com_indices.append([v_index[0][0], v_index[0][1]])
                else:
                    continue
            com_indices = np.asarray(com_indices, dtype='int32')
            com_indices = np.unique(com_indices, axis=0)

            # --------------------------------------------------------------------------------------------
            # Transforms the indices such that each pair of nuclei are associated with their common vertex
            t_indices = np.where(com_indices % 2 == 0, com_indices / 2, (com_indices - 1) / 2)
            t_com_indices = t_indices.astype(int)

            # -----------------------------------------------------------------------------------
            # dom contains the common vertex, the two adjacent nuclei and their average distances,
            # and the nucleus distance to big_nucleus
            pre_dom = []
            for t in range(0, len(t_com_indices)):
                pre_dom.append([ortho_facets[i][com_indices[t, 0]], self.ortho_nuclei[i][t_com_indices[t, 0]],
                                self.bound_distance[i][t_com_indices[t, 0]],
                                self.adjacent_average[i][t_com_indices[t, 0]],
                                self.ortho_nuclei[i][t_com_indices[t, 1]],
                                self.bound_distance[i][t_com_indices[t, 1]],
                                self.adjacent_average[i][t_com_indices[t, 1]]])

            pre_dom = np.array(pre_dom, dtype='int32')
            dom.append(pre_dom)

        # --------------------------------------------------------------------
        # Groups togheter all nuclei's centers and their area in a 1-2 fashion
        some_area = []
        for t in range(0, len(self.centers)):
            some_area.append(self.centers[t])

            if 20000 < self.area[t] < 25000:
                some_area.append([self.area[t] / 1, self.area[t] / 1])
            else:
                some_area.append([self.area[t], self.area[t]])

        some_area = np.asarray(some_area, dtype='int32')

        # -----------------------------------------------------------------------------------------
        # mod1 contains the common vertex, the two adjacent nuclei with their specific distance and
        # average distances and their areas, plus the big nuclei, their average distances
        # and their areas and their average distances
        for i in range(0, len(dom)):
            mod1 = []
            for g in dom[i]:
                mod1.append(g[0])
                for m in range(3, len(g), 3):
                    m_index = np.where(g[m - 2] == some_area)[0]
                    mod1.append(g[m - 2])
                    mod1.append(g[m - 1])
                    mod1.append(g[m])
                    mod1.append(some_area[m_index[0] + 1])
                mod1.append(self.bound_nuclei[i])
                mod1.append([9999, 9999])
                mod1.append(self.bound_nuclei_distance[i])
                mod1.append(big_areas[i])

            mod1 = np.asarray(mod1)
            mod1 = np.ceil(mod1).astype(int)
            mod1 = np.reshape(mod1, (-1, 13, 2))

            mod2.append(mod1)

            dd = np.empty((0, 2), dtype='int32')
            for q in mod1:

                area_1, adist_1, sdist_1 = math.sqrt(q[4, 0]), math.sqrt(q[3, 0]), math.sqrt(q[2, 0])
                area_2, adist_2, sdist_2 = math.sqrt(q[8, 0]), math.sqrt(q[7, 0]), math.sqrt(q[6, 0])
                area_bound, adist_bound = math.sqrt(q[12, 0]), math.sqrt(q[11, 0])

                total_sum = (1 / area_1) + (0.15 / adist_1) + (1 / area_2) + (0.15 / adist_2) + \
                            (1 / area_bound) + (0.15 / adist_bound)

                first_weight = ((1 / area_1) + (0.15 / adist_1)) / total_sum
                second_weight = ((1 / area_2) + (0.15 / adist_2)) / total_sum
                third_weight = ((1 / area_bound) + (0.15 / adist_bound)) / total_sum

                # ----------------------------------------------------
                # Calculates the lenght of each delaunay triangle side
                a2 = ((q[1, 0] - q[5, 0]) ** 2 + (q[1, 1] - q[5, 1]) ** 2)
                b2 = ((q[1, 0] - q[9, 0]) ** 2 + (q[1, 1] - q[9, 1]) ** 2)
                c2 = ((q[5, 0] - q[9, 0]) ** 2 + (q[5, 1] - q[9, 1]) ** 2)

                a = np.sqrt(np.absolute(a2))
                b = np.sqrt(np.absolute(b2))
                c = np.sqrt(np.absolute(c2))

                # ---------------------------------------------
                # Calculates the area of each delaunay triangle
                s = (a + b + c) / 2
                t_area = (s * (s - a) * (s - b) * (s - c)) ** 0.5

                # -----------------------------------------------
                # Calculates the angles of each delaunay triangle
                alpha = degrees(acos((b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c)))
                beta = degrees(acos((a ** 2 + c ** 2 - b ** 2) / (2.0 * a * c)))
                gamma = degrees(acos((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b)))

                d = q[1] * first_weight + q[5] * second_weight + q[9] * third_weight
                d = np.ceil(d).astype(int)

                angle = 10

                if (alpha > angle and beta > angle and gamma > angle) and t_area > 41000:
                    dd = np.append(dd, d)
                else:
                    dd = np.append(dd, q[0])

                # ----------------------------
                # Prints the modified vertices
                # cv.circle(self.print_img, (d[0], d[1]), 8, (245, 175, 14), -1, cv.LINE_AA)
                # cv.circle(self.resized, (d[0], d[1]), 8, (245, 175, 14), -1, cv.LINE_AA)

            dd = np.reshape(dd, (-1, 2))
            dd2.append(dd)

        last_facets = []
        for k in range(0, len(self.facets)):
            f_facet = np.empty((0, 0), dtype='int32')
            f_facet = np.append(f_facet, np.floor(self.facets[k]).astype(int))
            f_facet = np.reshape(f_facet, (-1, 2))

            for y in range(0, len(mod2)):
                dime = np.array(dd2[y], dtype='int32')
                for g in range(0, len(mod2[y])):
                    f_index = np.where((f_facet[:, 0] == mod2[y][g, 0, 0]) &
                                       (f_facet[:, 1] == mod2[y][g, 0, 1]))
                    f_index = np.unique(f_index[0])

                    if f_index.size > 0:
                        f_facet[f_index] = dime[g]
                    else:
                        continue

            last_facets.append(f_facet)

        for m_facet in last_facets:
            cv.polylines(self.resized, [m_facet], True, (240, 240, 240), 3, cv.LINE_AA)
            cv.polylines(self.print_img, [m_facet], True, (240, 240, 240), 3, cv.LINE_AA)

        for s in last_facets:
            for dot in s:
                cv.circle(self.print_img, (dot[0], dot[1]), 8, (0, 105, 217), -1, cv.LINE_AA)
                cv.circle(self.resized, (dot[0], dot[1]), 8, (0, 105, 217), -1, cv.LINE_AA)

        cv.imshow(self.name, self.resized)

        while True:
            k = cv.waitKey(0) & 0xFF
            if k == ord('j'):
                cv.imshow(self.name, self.resized)
            elif k == ord('l'):
                cv.imshow(self.name, self.print_img)
            else:
                cv.destroyAllWindows()
                break


start = ReshapeVoronoi(file_path)


def start_all():
    start.add_image()
    start.load_image()
    start.contour_detection()
    edge_list = start.get_facets()
    start.distances(edge_list)
    big_areas = start.find_orthogonal_facets()
    ortho_facets = start.differences()
    start.barycentric_coordinates(ortho_facets, big_areas)

if __name__ == "__main__":
    start_all()

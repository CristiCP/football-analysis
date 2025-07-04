from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dic = {}
        self.kmeans = None

    def get_top_cluster(self, image):
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_image = image[:int(image.shape[0] / 2), :]

        kmeans = self.get_top_cluster(top_half_image)

        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0],
                          clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        if len(player_colors) < 2:
            print("[WARNING] Not enough player colors for KMeans clustering — skipping team color assignment.")
            self.kmeans = None
            return
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if self.kmeans is None:
            print(f"[WARNING] Skipping team assignment for player {player_id} — KMeans not initialized.")
            return 0

        if player_id in self.player_team_dic:
            return self.player_team_dic[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dic[player_id] = team_id

        return team_id

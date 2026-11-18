"""Folium map generation (EDA and prediction maps)."""
import numpy as np
import folium


def _arc_lines(nodes, arcs, ids):
    return [[(nodes[a, 1], nodes[a, 0]), (nodes[b, 1], nodes[b, 0])] for a, b in arcs[ids]]


def trip_map(trip, pred_route=None, title="", network_pad_deg=0.002, max_network_arcs=20000):
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    centre = [float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))]
    # CartoDB/OSM tiles can be blocked ("API key" / "Access blocked"); Esri tiles load from file:// too
    m = folium.Map(location=centre, zoom_start=14, tiles=None)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(esri.format("Canvas/World_Light_Gray_Base"), attr="Tiles &copy; Esri", name="Esri light grey").add_to(m)
    folium.TileLayer(esri.format("World_Street_Map"), attr="Tiles &copy; Esri", name="Esri streets", show=False).add_to(m)

    # road network near the trajectory
    lo, hi = tr[:, :2].min(0) - network_pad_deg, tr[:, :2].max(0) + network_pad_deg
    inside = ((nd >= lo) & (nd <= hi)).all(1)
    sel = np.flatnonzero(inside[ar[:, 0]] & inside[ar[:, 1]])[:max_network_arcs]
    fg_net = folium.FeatureGroup(name="Road network (nearby)", show=True)
    folium.PolyLine(_arc_lines(nd, ar, sel), color="#9e9e9e", weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    if trip.get("route") is not None:
        fg_ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_lines(nd, ar, trip["route"]), color="#2e7d32", weight=6, opacity=0.7,
                        tooltip="reference route").add_to(fg_ref)
        fg_ref.add_to(m)

    if pred_route is not None and len(pred_route):
        fg_pred = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_arc_lines(nd, ar, np.asarray(pred_route)), color="#d32f2f", weight=3,
                        opacity=0.9, tooltip="predicted route").add_to(fg_pred)
        fg_pred.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS points", show=True)
    step = max(1, len(tr) // 1500)
    for lon, lat, t in tr[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True, weight=0,
                            fill_opacity=0.8, tooltip=f"t={t:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
    padding:8px 12px;border:1px solid #999;font-size:13px">
    <b>{title}</b><br><span style="color:#1565c0">&#9679;</span> raw GPS<br>
    <span style="color:#9e9e9e">&#9473;</span> road network<br>
    <span style="color:#2e7d32">&#9473;</span> reference route<br>
    <span style="color:#d32f2f">&#9473;</span> predicted route</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    return m

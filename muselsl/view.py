
def view(window=5, scale=100, refresh=0.2, figure="15x6", version=1, backend='TkAgg', data_source="EEG"):
    print(f"Starting viewer for {data_source} data, version {version}")
    if version == 2:
        from . import viewer_v2
        viewer_v2.view()
    else:
        from . import viewer_v1
        print(f"Using viewer_v1 for {data_source} data")
        viewer_v1.view(window, scale, refresh, figure, backend, data_source)

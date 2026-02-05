
import json
import networkx as nx

def nx_to_sigma_json(G: nx.Graph, df=None) -> dict: return {"nodes": [], "edges": []}

G = nx.Graph()
sigma_json = json.dumps(nx_to_sigma_json(G))

# Copying the f-string from dashboard_probable.py logic
sigma_html = f"""
<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8">
</head>
<body>
    <script>
        async function init() {{
            try {{
                const Library = {{ layoutForceAtlas2: {{}} }};
                const FA2 = Library.layoutForceAtlas2;
                const nodeCount = 1000;
                let isTurbo = true;
                let initialGravity = 0.01;
                let initialScaling = 200000;
                let initialTheta = 0.8;
                let layoutInstance = null;
                let isManualLoop = false;
                let isPaused = false;
                let fa2Settings = {{
                    gravity: initialGravity,
                    scalingRatio: initialScaling,
                    adjustSizes: true,
                    outboundAttractionDistribution: true, 
                    linLogMode: false,
                    barnesHutOptimize: true,
                    barnesHutTheta: initialTheta,
                    strongGravityMode: false 
                }};

                if (isTurbo) {{
                    // dynamic status
                }} else {{
                    // mode precision
                }}

                function startContinuousLayout() {{
                    try {{
                        const Constructor = null;
                        if (Constructor) {{
                            layoutInstance = new Constructor();
                        }} else {{
                            throw new Error("No FA2 Constructor");
                        }}
                    }} catch (e) {{
                        isManualLoop = true;
                    }}
                }}

                function runManualStep() {{
                    if (isPaused || !isManualLoop) return;
                    requestAnimationFrame(runManualStep);
                }}

                function toggle() {{
                    isPaused = !isPaused;
                }}

                if (true) startContinuousLayout();
            }} catch (err) {{
                console.error(err);
            }}
        }}
        init();
    </script>
</body>
</html>
"""
print(sigma_html)

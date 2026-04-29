using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class SetupScene
{
    [MenuItem("Tools/Build CartPole Scenes")]
    public static void CreateAllScenes()
    {
        CreateArenaScene();
        CreateMenuScene();
    }

    public static void CreateArenaScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var lightObj = new GameObject("Directional Light");
        var light = lightObj.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.1f;
        lightObj.transform.rotation = Quaternion.Euler(45f, -35f, 0f);

        var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
        ground.name = "Ground";
        ground.transform.position = new Vector3(0f, 0f, 0f);
        ground.transform.localScale = new Vector3(12f, 0.1f, 6f);

        var lineTop = GameObject.CreatePrimitive(PrimitiveType.Cube);
        lineTop.transform.position = new Vector3(0f, 0.051f, 1.2f);
        lineTop.transform.localScale = new Vector3(12f, 0.01f, 0.05f);

        var lineBottom = GameObject.CreatePrimitive(PrimitiveType.Cube);
        lineBottom.transform.position = new Vector3(0f, 0.051f, -1.2f);
        lineBottom.transform.localScale = new Vector3(12f, 0.01f, 0.05f);

        var cartDqn = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cartDqn.name = "Cart_DQN";
        cartDqn.transform.position = new Vector3(0f, 0.5f, 1.2f);
        cartDqn.transform.localScale = new Vector3(0.8f, 0.4f, 0.4f);

        var poleDqn = GameObject.CreatePrimitive(PrimitiveType.Cube);
        poleDqn.name = "Pole_DQN";
        poleDqn.transform.position = new Vector3(0f, 1.0f, 1.2f);
        poleDqn.transform.localScale = new Vector3(0.12f, 1.2f, 0.12f);

        var cartRnd = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cartRnd.name = "Cart_Random";
        cartRnd.transform.position = new Vector3(0f, 0.5f, -1.2f);
        cartRnd.transform.localScale = new Vector3(0.8f, 0.4f, 0.4f);

        var poleRnd = GameObject.CreatePrimitive(PrimitiveType.Cube);
        poleRnd.name = "Pole_Random";
        poleRnd.transform.position = new Vector3(0f, 1.0f, -1.2f);
        poleRnd.transform.localScale = new Vector3(0.12f, 1.2f, 0.12f);

        CreateWheel(cartDqn.transform, new Vector3(-0.25f, -0.25f, 0.22f));
        CreateWheel(cartDqn.transform, new Vector3(0.25f, -0.25f, 0.22f));
        CreateWheel(cartRnd.transform, new Vector3(-0.25f, -0.25f, -0.22f));
        CreateWheel(cartRnd.transform, new Vector3(0.25f, -0.25f, -0.22f));

        var root = new GameObject("ArenaRoot");
        var dqn = root.AddComponent<DQNInference>();
        var manager = root.AddComponent<CartPoleArenaManager>();
        var capture = root.AddComponent<ReplayCapture>();
        capture.captureEnabled = false;
        var exporter = root.AddComponent<BenchmarkExporter>();
        var hud = root.AddComponent<ArenaHUD>();

        var dqnAgentObj = new GameObject("DQNAgent");
        var dqnAgent = dqnAgentObj.AddComponent<CartPoleAgent>();
        dqnAgent.agentName = "DQN";
        dqnAgent.useRandomPolicy = false;
        dqnAgent.cart = cartDqn.transform;
        dqnAgent.pole = poleDqn.transform;
        dqnAgent.dqn = dqn;

        var rndAgentObj = new GameObject("RandomAgent");
        var rndAgent = rndAgentObj.AddComponent<CartPoleAgent>();
        rndAgent.agentName = "Random";
        rndAgent.useRandomPolicy = true;
        rndAgent.cart = cartRnd.transform;
        rndAgent.pole = poleRnd.transform;
        rndAgent.dqn = dqn;

        manager.dqnAgent = dqnAgent;
        manager.randomAgent = rndAgent;

        exporter.manager = manager;
        exporter.capture = capture;

        hud.manager = manager;
        hud.dqn = dqnAgent;
        hud.random = rndAgent;

        var camLeft = new GameObject("Camera_DQN");
        var c1 = camLeft.AddComponent<Camera>();
        c1.rect = new Rect(0f, 0f, 0.5f, 1f);
        c1.clearFlags = CameraClearFlags.Skybox;
        camLeft.tag = "MainCamera";
        var f1 = camLeft.AddComponent<CinematicFollowCamera>();
        f1.target = cartDqn.transform;
        f1.offset = new Vector3(0f, 2.8f, -7.5f);
        var s1 = camLeft.AddComponent<CameraShake>();
        s1.sourceAgent = dqnAgent;

        var camRight = new GameObject("Camera_Random");
        var c2 = camRight.AddComponent<Camera>();
        c2.rect = new Rect(0.5f, 0f, 0.5f, 1f);
        c2.clearFlags = CameraClearFlags.Skybox;
        var f2 = camRight.AddComponent<CinematicFollowCamera>();
        f2.target = cartRnd.transform;
        f2.offset = new Vector3(0f, 2.8f, -7.5f);
        var s2 = camRight.AddComponent<CameraShake>();
        s2.sourceAgent = rndAgent;

        System.IO.Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, "Assets/Scenes/CartPole.unity");
    }

    public static void CreateMenuScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var cam = new GameObject("MenuCamera");
        cam.tag = "MainCamera";
        cam.AddComponent<Camera>();

        var menuCtrlObj = new GameObject("MenuController");
        menuCtrlObj.AddComponent<MenuController>();

        System.IO.Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, "Assets/Scenes/Menu.unity");

        EditorBuildSettings.scenes = new[] {
            new EditorBuildSettingsScene("Assets/Scenes/Menu.unity", true),
            new EditorBuildSettingsScene("Assets/Scenes/CartPole.unity", true)
        };
    }

    static void CreateWheel(Transform parent, Vector3 localPos)
    {
        var w = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        w.transform.SetParent(parent, false);
        w.transform.localScale = new Vector3(0.16f, 0.06f, 0.16f);
        w.transform.localPosition = localPos;
        w.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
        var spin = w.AddComponent<WheelSpin>();
        spin.cart = parent;
    }
}

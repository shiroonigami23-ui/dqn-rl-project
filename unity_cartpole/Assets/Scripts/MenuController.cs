using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuController : MonoBehaviour
{
    public string sceneToLoad = "CartPole";

    void OnGUI()
    {
        GUIStyle title = new GUIStyle(GUI.skin.label);
        title.fontSize = 44;
        title.alignment = TextAnchor.MiddleCenter;
        title.normal.textColor = Color.white;

        GUIStyle sub = new GUIStyle(GUI.skin.label);
        sub.fontSize = 20;
        sub.alignment = TextAnchor.MiddleCenter;
        sub.normal.textColor = new Color(0.85f, 0.9f, 1f);

        GUIStyle btn = new GUIStyle(GUI.skin.button);
        btn.fontSize = 24;

        float cx = Screen.width * 0.5f;

        GUI.Label(new Rect(cx - 420, 110, 840, 70), "DQN CartPole Arena", title);
        GUI.Label(new Rect(cx - 520, 180, 1040, 40), "Split-screen DQN vs Random with cinematic camera + replay export", sub);

        if (GUI.Button(new Rect(cx - 140, 280, 280, 70), "Start Arena", btn))
        {
            SceneManager.LoadScene(sceneToLoad);
        }

        if (GUI.Button(new Rect(cx - 140, 370, 280, 70), "Quit", btn))
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
}

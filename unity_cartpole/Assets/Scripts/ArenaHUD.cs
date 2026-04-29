using UnityEngine;

public class ArenaHUD : MonoBehaviour
{
    public CartPoleArenaManager manager;
    public CartPoleAgent dqn;
    public CartPoleAgent random;

    void OnGUI()
    {
        GUIStyle box = new GUIStyle(GUI.skin.box);
        box.fontSize = 16;
        box.alignment = TextAnchor.UpperLeft;

        if (manager != null)
            GUI.Box(new Rect(Screen.width / 2 - 140, 10, 280, 35), $"Episode {manager.episodeIndex}  |  R reset", box);

        if (dqn != null)
        {
            string q = (dqn.LastQ != null && dqn.LastQ.Length == 2) ? $"Q=[{dqn.LastQ[0]:F2},{dqn.LastQ[1]:F2}]" : "Q=[?,?]";
            GUI.Box(new Rect(10, 10, 380, 110),
                $"DQN\nStep {dqn.StepCount}  Reward {dqn.EpisodeReward:F1}\nx={dqn.X:F3}  theta={dqn.ThetaDeg:F2}°\n{q}", box);
        }

        if (random != null)
        {
            GUI.Box(new Rect(Screen.width - 390, 10, 380, 110),
                $"Random\nStep {random.StepCount}  Reward {random.EpisodeReward:F1}\nx={random.X:F3}  theta={random.ThetaDeg:F2}°", box);
        }
    }
}

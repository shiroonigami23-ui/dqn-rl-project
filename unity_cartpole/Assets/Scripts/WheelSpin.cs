using UnityEngine;

public class WheelSpin : MonoBehaviour
{
    public Transform cart;
    public float wheelRadius = 0.15f;

    private float lastX;

    void Start()
    {
        if (cart != null) lastX = cart.position.x;
    }

    void Update()
    {
        if (cart == null) return;
        float dx = cart.position.x - lastX;
        float angle = -(dx / Mathf.Max(0.001f, wheelRadius)) * Mathf.Rad2Deg;
        transform.Rotate(angle, 0f, 0f, Space.Self);
        lastX = cart.position.x;
    }
}

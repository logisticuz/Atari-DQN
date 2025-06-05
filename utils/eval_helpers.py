def save_video_and_snapshots_from_env(env, model, video_path, snapshot_dir="snapshots", csv_path="reward_log.csv",
                                      episodes=1, overlay_flags=None, force_left=False):
    video_logger = VideoLogger(output_path=video_path, overlay_flags=overlay_flags)
    event_logger = EventLogger(snapshot_dir=snapshot_dir, csv_path=csv_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            action = 1 if force_left else model.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaping_dict = info.get("shaping", {})
            died = info.get("died", False)

            video_logger.log_frame(obs)
            event_logger.log_step(step=step, obs=obs, action=action,
                                  shaping_dict=shaping_dict, died=died)

            obs = next_obs
            step += 1

    video_logger.save()
    event_logger.save()

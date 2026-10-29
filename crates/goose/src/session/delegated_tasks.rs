use super::SessionManager;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::Row;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DelegatedTaskEvent {
    pub task_id: String,
    pub parent_session_id: String,
    pub description: String,
    pub status: String,
}

impl SessionManager {
    pub(crate) async fn record_delegated_task(
        &self,
        event: &DelegatedTaskEvent,
        output: &str,
    ) -> Result<()> {
        sqlx::query("INSERT OR IGNORE INTO delegated_task_outbox (task_id, parent_session_id, description, status, output) VALUES (?, ?, ?, ?, ?)")
            .bind(&event.task_id).bind(&event.parent_session_id).bind(&event.description).bind(&event.status).bind(output)
            .execute(self.storage().pool().await?).await?;
        Ok(())
    }

    pub(crate) async fn delegated_task_result(
        &self,
        parent: &str,
        task: &str,
        consume: bool,
    ) -> Result<Option<(String, String, String)>> {
        let mut tx = self.storage().pool().await?.begin().await?;
        let row = sqlx::query("SELECT description, status, output FROM delegated_task_outbox WHERE task_id = ? AND parent_session_id = ? AND result_consumed = 0")
            .bind(task).bind(parent).fetch_optional(&mut *tx).await?;
        let result = row
            .map(|row| {
                Ok::<_, sqlx::Error>((
                    row.try_get("description")?,
                    row.try_get("status")?,
                    row.try_get("output")?,
                ))
            })
            .transpose()?;
        if consume && result.is_some() {
            sqlx::query("UPDATE delegated_task_outbox SET result_consumed = 1 WHERE task_id = ? AND parent_session_id = ?")
                .bind(task).bind(parent).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(result)
    }

    pub(crate) async fn pending_delegated_tasks(
        &self,
        parents: &[String],
    ) -> Result<Vec<DelegatedTaskEvent>> {
        if parents.is_empty() {
            return Ok(Vec::new());
        }
        let mut query = sqlx::QueryBuilder::<sqlx::Sqlite>::new("SELECT task_id, parent_session_id, description, status FROM delegated_task_outbox WHERE acknowledged = 0 AND parent_session_id IN (");
        let mut ids = query.separated(", ");
        for parent in parents {
            ids.push_bind(parent);
        }
        ids.push_unseparated(") ORDER BY rowid LIMIT 1000");
        let rows = query
            .build()
            .fetch_all(self.storage().pool().await?)
            .await?;
        let mut events = Vec::new();
        for row in rows {
            let parent_session_id: String = row.try_get("parent_session_id")?;
            if parents.contains(&parent_session_id) {
                events.push(DelegatedTaskEvent {
                    task_id: row.try_get("task_id")?,
                    parent_session_id,
                    description: row.try_get("description")?,
                    status: row.try_get("status")?,
                });
            }
        }
        Ok(events)
    }

    pub(crate) async fn acknowledge_delegated_task(&self, parent: &str, task: &str) -> Result<()> {
        sqlx::query("UPDATE delegated_task_outbox SET acknowledged = 1 WHERE task_id = ? AND parent_session_id = ?")
            .bind(task).bind(parent).execute(self.storage().pool().await?).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn delegated_outbox_replays_until_owner_ack_and_retains_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let manager = SessionManager::new(directory.path().to_path_buf());
        let event = DelegatedTaskEvent {
            task_id: "task".into(),
            parent_session_id: "owner".into(),
            description: "work".into(),
            status: "action_limit_reached".into(),
        };
        manager
            .record_delegated_task(&event, "checkpoint")
            .await
            .unwrap();
        manager
            .record_delegated_task(&event, "duplicate")
            .await
            .unwrap();
        drop(manager);
        let manager = SessionManager::new(directory.path().to_path_buf());
        assert!(manager
            .pending_delegated_tasks(&["other".into()])
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            manager
                .pending_delegated_tasks(&["owner".into()])
                .await
                .unwrap()
                .len(),
            1
        );
        manager
            .acknowledge_delegated_task("other", "task")
            .await
            .unwrap();
        assert_eq!(
            manager
                .pending_delegated_tasks(&["owner".into()])
                .await
                .unwrap()
                .len(),
            1
        );
        manager
            .acknowledge_delegated_task("owner", "task")
            .await
            .unwrap();
        assert!(manager
            .pending_delegated_tasks(&["owner".into()])
            .await
            .unwrap()
            .is_empty());
        let output: String =
            sqlx::query_scalar("SELECT output FROM delegated_task_outbox WHERE task_id = ?")
                .bind("task")
                .fetch_one(manager.storage().pool().await.unwrap())
                .await
                .unwrap();
        assert_eq!(
            output, "checkpoint",
            "acknowledgement must not consume worker output"
        );
        assert!(manager
            .delegated_task_result("other", "task", false)
            .await
            .unwrap()
            .is_none());
        let (_, _, checkpoint) = manager
            .delegated_task_result("owner", "task", false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint, "checkpoint");
        assert!(manager
            .delegated_task_result("owner", "task", true)
            .await
            .unwrap()
            .is_some());
        assert!(manager
            .delegated_task_result("owner", "task", false)
            .await
            .unwrap()
            .is_none());
    }
}

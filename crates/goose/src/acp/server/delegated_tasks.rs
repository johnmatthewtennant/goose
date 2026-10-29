use super::*;

impl GooseAcpAgent {
    pub(super) async fn dispatch_delegated_task_outbox(
        &self,
        action: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, agent_client_protocol::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Pending {
            parent_session_ids: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Acknowledge {
            parent_session_id: String,
            task_id: String,
        }
        match action {
            "pending" => {
                let args: Pending = serde_json::from_value(params).map_err(|error| {
                    agent_client_protocol::Error::invalid_params().data(error.to_string())
                })?;
                if args.parent_session_ids.len() > 1000 {
                    return Err(agent_client_protocol::Error::invalid_params()
                        .data("Too many parent sessions"));
                }
                let events = self
                    .session_manager
                    .pending_delegated_tasks(&args.parent_session_ids)
                    .await
                    .internal_err_ctx("Failed to read delegated task outbox")?;
                Ok(serde_json::json!({ "events": events }))
            }
            "acknowledge" => {
                let args: Acknowledge = serde_json::from_value(params).map_err(|error| {
                    agent_client_protocol::Error::invalid_params().data(error.to_string())
                })?;
                self.session_manager
                    .acknowledge_delegated_task(&args.parent_session_id, &args.task_id)
                    .await
                    .internal_err_ctx("Failed to acknowledge delegated task delivery")?;
                Ok(serde_json::json!({}))
            }
            _ => Err(agent_client_protocol::Error::method_not_found()),
        }
    }
}

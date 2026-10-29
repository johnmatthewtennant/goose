use crate::{
    agents::{subagent_task_config::TaskConfig, Agent, AgentConfig, AgentEvent, SessionConfig},
    conversation::{
        message::{Message, MessageContent},
        Conversation,
    },
    prompt_template::render_template,
    recipe::Recipe,
};
use anyhow::{anyhow, Result};
use futures::StreamExt;
use rmcp::model::{ErrorCode, ErrorData, Notification, ServerNotification};
#[expect(deprecated)]
use rmcp::model::{LoggingLevel, LoggingMessageNotificationParam};
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

pub type OnMessageCallback = Arc<dyn Fn(&Message) + Send + Sync>;

#[derive(Serialize)]
pub struct SubagentPromptContext {
    pub max_turns: usize,
    pub task_instructions: String,
    pub tool_count: usize,
    pub available_tools: String,
}

type AgentMessagesFuture =
    Pin<Box<dyn Future<Output = Result<(Conversation, Option<String>)>> + Send>>;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub task_config: TaskConfig,
    pub return_last_only: bool,
    pub session_id: String,
    pub cancellation_token: Option<CancellationToken>,
    pub on_message: Option<OnMessageCallback>,
    pub notification_tx: Option<tokio::sync::mpsc::UnboundedSender<ServerNotification>>,
}

pub async fn run_subagent_task(params: SubagentRunParams) -> Result<String, anyhow::Error> {
    let return_last_only = params.return_last_only;
    let (messages, final_output) = get_agent_messages(params).await.map_err(|e| {
        ErrorData::new(
            ErrorCode::INTERNAL_ERROR,
            format!("Failed to execute task: {}", e),
            None,
        )
    })?;

    if let Some(output) = final_output {
        return Ok(output);
    }

    Ok(extract_response_text(&messages, return_last_only))
}

fn extract_response_text(messages: &Conversation, return_last_only: bool) -> String {
    if return_last_only {
        messages
            .messages()
            .last()
            .and_then(|message| {
                message.content.iter().find_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| String::from("No text content in last message"))
    } else {
        let all_text_content: Vec<String> = messages
            .iter()
            .flat_map(|message| {
                message.content.iter().filter_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    crate::conversation::message::MessageContent::ToolResponse(tool_response) => {
                        if let Ok(result) = &tool_response.tool_result {
                            let texts: Vec<String> = result
                                .content
                                .iter()
                                .filter_map(|content| {
                                    if let rmcp::model::ContentBlock::Text(raw_text_content) =
                                        content
                                    {
                                        Some(raw_text_content.text.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if !texts.is_empty() {
                                Some(format!("Tool result: {}", texts.join("\n")))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
            })
            .collect();

        all_text_content.join("\n")
    }
}

pub const SUBAGENT_TOOL_REQUEST_TYPE: &str = "subagent_tool_request";

fn get_agent_messages(params: SubagentRunParams) -> AgentMessagesFuture {
    Box::pin(async move {
        let SubagentRunParams {
            config,
            recipe,
            task_config,
            session_id,
            cancellation_token,
            on_message,
            notification_tx,
            ..
        } = params;

        let system_instructions = recipe.instructions.clone().unwrap_or_default();
        let user_task = recipe
            .prompt
            .clone()
            .unwrap_or_else(|| "Begin.".to_string());

        let session_manager = Arc::clone(&config.session_manager);
        let agent = Arc::new(Agent::with_config(config));

        agent
            .update_provider(
                task_config.provider.clone(),
                task_config.model_config.clone(),
                &session_id,
            )
            .await
            .map_err(|e| anyhow!("Failed to set provider on sub agent: {}", e))?;

        for extension in &task_config.extensions {
            if let Err(e) = agent.add_extension(extension.clone(), &session_id).await {
                debug!(
                    "Failed to add extension '{}' to subagent: {}",
                    extension.name(),
                    e
                );
            }
        }

        let has_response_schema = recipe.response.is_some();
        agent
            .apply_recipe_components(recipe.response.clone(), true)
            .await?;

        let max_turns = task_config
            .max_turns
            .expect("TaskConfig always sets max_turns");
        let subagent_prompt =
            build_subagent_prompt(&agent, max_turns, &session_id, system_instructions).await?;
        agent.override_system_prompt(subagent_prompt).await;

        let user_message =
            Message::user().with_text(format!("Subagent ID: {session_id}\n\n{user_task}"));
        let mut conversation = Conversation::new_unvalidated(vec![user_message.clone()]);

        agent
            .config
            .session_manager
            .update(&session_id)
            .recipe(Some(recipe.clone()))
            .apply()
            .await?;

        if let Some(activities) = recipe.activities {
            for activity in activities {
                info!("Recipe activity: {}", activity);
            }
        }
        let session_config = SessionConfig {
            id: session_id.clone(),
            schedule_id: None,
            max_turns: task_config.max_turns.map(|v| v as u32),
            retry_config: recipe.retry,
        };

        let mut stream =
            crate::session_context::with_session_id(Some(session_id.to_string()), async {
                agent
                    .reply(
                        user_message,
                        session_config,
                        crate::agents::state_machine::enabled(),
                        cancellation_token,
                    )
                    .await
            })
            .await
            .map_err(|e| anyhow!("Failed to get reply from agent: {}", e))?;

        collect_subagent_stream(
            &mut stream,
            &mut conversation,
            &session_id,
            on_message.as_ref(),
            notification_tx.as_ref(),
        )
        .await?;

        // Streaming messages can omit persisted errors or include an error that
        // a Stop hook, steer, or recipe retry subsequently recovers from.
        let saved = session_manager
            .get_session(&session_id, true)
            .await
            .map_err(|error| {
                subagent_failure(
                    &format!("Failed to read final worker outcome: {error}"),
                    &conversation,
                )
            })?;
        let saved = saved
            .conversation
            .as_ref()
            .ok_or_else(|| subagent_failure("Missing final worker conversation", &conversation))?;
        check_subagent_outcome(&conversation, saved)?;
        let final_output = get_final_output(&agent, has_response_schema).await;

        Ok((conversation, final_output))
    })
}

async fn collect_subagent_stream(
    stream: &mut futures::stream::BoxStream<'_, Result<AgentEvent>>,
    conversation: &mut Conversation,
    session_id: &str,
    on_message: Option<&OnMessageCallback>,
    notification_tx: Option<&tokio::sync::mpsc::UnboundedSender<ServerNotification>>,
) -> Result<()> {
    while let Some(message_result) = stream.next().await {
        match message_result {
            Ok(AgentEvent::Message(msg)) => {
                if let Some(callback) = on_message {
                    callback(&msg);
                }
                if let Some(tx) = notification_tx {
                    for content in &msg.content {
                        if let Some(notif) = create_tool_notification(content, &session_id) {
                            if tx.send(notif).is_err() {
                                debug!("Notification receiver dropped for subagent {}", session_id);
                            }
                        }
                    }
                }
                conversation.push(msg);
            }
            Ok(AgentEvent::Usage(_)) => {}
            Ok(AgentEvent::MessageUsage { .. }) => {}
            Ok(AgentEvent::McpNotification(_)) => {}
            Ok(AgentEvent::HistoryReplaced(updated_conversation)) => {
                *conversation = updated_conversation;
            }
            Err(e) => {
                return Err(subagent_failure(
                    &format!("Agent response stream failed: {e}"),
                    &conversation,
                ));
            }
        }
    }
    Ok(())
}

fn subagent_failure(reason: &str, conversation: &Conversation) -> anyhow::Error {
    anyhow!(
        "{reason}\n\n## Worker progress\n\n{}",
        extract_response_text(conversation, false)
    )
}

fn check_subagent_outcome(observed: &Conversation, saved: &Conversation) -> Result<()> {
    use crate::conversation::message::{MessageErrorKind, SystemNotificationType};
    // Authentication and credit exhaustion remain explicit, even when another
    // operation emits text later. Operational errors use the final saved turn.
    for content in observed
        .messages()
        .iter()
        .flat_map(|message| &message.content)
    {
        match content {
            MessageContent::Error(error)
                if matches!(
                    error.kind,
                    MessageErrorKind::Authentication | MessageErrorKind::CreditsExhausted
                ) =>
            {
                return Err(subagent_failure(
                    &format!("{:?}: {}", error.kind, error.message),
                    saved,
                ));
            }
            MessageContent::SystemNotification(notice)
                if notice.notification_type == SystemNotificationType::CreditsExhausted =>
            {
                return Err(subagent_failure(
                    &format!("CreditsExhausted: {}", notice.msg),
                    saved,
                ));
            }
            _ => {}
        }
    }
    if let Some(error) = saved.terminal_error() {
        return Err(subagent_failure(
            &format!("{:?}: {}", error.kind, error.message),
            saved,
        ));
    }
    Ok(())
}

async fn build_subagent_prompt(
    agent: &Agent,
    max_turns: usize,
    session_id: &str,
    system_instructions: String,
) -> Result<String> {
    let mut tool_names: Vec<_> = agent
        .list_tools(session_id, None)
        .await
        .into_iter()
        .filter(super::reply_parts::is_tool_visible_to_model)
        .map(|t| t.name.to_string())
        .collect();
    tool_names.sort_unstable();
    render_template(
        "subagent_system.md",
        &SubagentPromptContext {
            max_turns,
            task_instructions: system_instructions,
            tool_count: tool_names.len(),
            available_tools: tool_names.join(", "),
        },
    )
    .map_err(|e| anyhow!("Failed to render subagent system prompt: {}", e))
}

async fn get_final_output(agent: &Agent, has_response_schema: bool) -> Option<String> {
    if has_response_schema {
        agent
            .final_output_tool
            .lock()
            .await
            .as_ref()
            .and_then(|tool| tool.final_output.clone())
    } else {
        None
    }
}

#[expect(deprecated)]
pub fn create_tool_notification(
    content: &MessageContent,
    subagent_id: &str,
) -> Option<ServerNotification> {
    if let MessageContent::ToolRequest(req) = content {
        let tool_call = req.tool_call.as_ref().ok()?;

        Some(ServerNotification::LoggingMessageNotification(
            Notification::new(
                LoggingMessageNotificationParam::new(
                    LoggingLevel::Info,
                    serde_json::json!({
                        "type": SUBAGENT_TOOL_REQUEST_TYPE,
                        "subagent_id": subagent_id,
                        "tool_call": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments
                        }
                    }),
                )
                .with_logger(format!("subagent:{}", subagent_id)),
            ),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{create_tool_notification, SUBAGENT_TOOL_REQUEST_TYPE};
    use crate::conversation::message::MessageContent;
    use rmcp::model::{CallToolRequestParams, ServerNotification};
    use serde_json::json;

    #[test]
    #[expect(deprecated)]
    fn create_tool_notification_for_tool_request() {
        let tool_call = CallToolRequestParams::new("developer__shell".to_string())
            .with_arguments(json!({"command": "ls"}).as_object().unwrap().clone());
        let content = MessageContent::tool_request("req1", Ok(tool_call));
        let notification =
            create_tool_notification(&content, "session_1").expect("expected notification");

        let ServerNotification::LoggingMessageNotification(log_notif) = notification else {
            panic!("expected logging notification");
        };
        let data = log_notif
            .params
            .data
            .as_object()
            .expect("expected object data");
        assert_eq!(
            data.get("type").and_then(|v| v.as_str()),
            Some(SUBAGENT_TOOL_REQUEST_TYPE)
        );
        assert_eq!(
            data.get("subagent_id").and_then(|v| v.as_str()),
            Some("session_1")
        );
        let tool_call = data
            .get("tool_call")
            .and_then(|v| v.as_object())
            .expect("expected tool_call object");
        assert_eq!(
            tool_call.get("name").and_then(|v| v.as_str()),
            Some("developer__shell")
        );
    }

    #[test]
    fn create_tool_notification_ignores_non_tool_request() {
        let content = MessageContent::text("hello");
        assert!(create_tool_notification(&content, "session_1").is_none());
    }
}

#[cfg(test)]
mod terminal_outcome_tests {
    use super::*;
    use crate::conversation::message::{MessageErrorKind, SystemNotificationType};
    use futures::stream;

    #[tokio::test]
    async fn stream_error_cannot_become_success_and_retains_progress() {
        let mut conversation = Conversation::empty();
        let mut stream: futures::stream::BoxStream<'_, Result<AgentEvent>> =
            Box::pin(stream::iter([
                Ok(AgentEvent::Message(
                    Message::assistant().with_text("partial checkpoint"),
                )),
                Err(anyhow!("transport disconnected")),
            ]));
        let error = collect_subagent_stream(&mut stream, &mut conversation, "worker", None, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("transport disconnected"));
        assert!(error.contains("partial checkpoint"));
    }

    #[test]
    fn saved_final_outcome_allows_recovery_but_not_notice_only_masking() {
        let observed = Conversation::new_unvalidated([
            Message::user().with_text("task"),
            Message::assistant().with_text("partial checkpoint"),
            Message::assistant().with_error(MessageErrorKind::Other, "provider failed"),
        ]);
        let mut saved = observed.clone();
        saved.push(Message::assistant().with_system_notification(
            SystemNotificationType::InlineMessage,
            "Stop hook cap reached",
        ));
        let error = check_subagent_outcome(&observed, &saved)
            .unwrap_err()
            .to_string();
        assert!(error.contains("provider failed"));
        assert!(error.contains("partial checkpoint"));
        saved.push(Message::assistant().with_text("recovered"));
        assert!(check_subagent_outcome(&observed, &saved).is_ok());
    }

    #[test]
    fn authentication_and_credits_remain_explicit_even_after_text() {
        for kind in [
            MessageErrorKind::Authentication,
            MessageErrorKind::CreditsExhausted,
        ] {
            let mut saved = Conversation::new_unvalidated([
                Message::user().with_text("task"),
                Message::assistant().with_error(kind, "blocked"),
            ]);
            saved.push(Message::assistant().with_text("later response"));
            assert!(check_subagent_outcome(&saved, &saved)
                .unwrap_err()
                .to_string()
                .contains(&format!("{kind:?}")));
        }
        let saved = Conversation::new_unvalidated([
            Message::user().with_text("task"),
            Message::assistant().with_system_notification(
                SystemNotificationType::CreditsExhausted,
                "top up credits",
            ),
        ]);
        assert!(check_subagent_outcome(&saved, &saved)
            .unwrap_err()
            .to_string()
            .contains("CreditsExhausted"));
    }
}

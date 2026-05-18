from app.models.conversation import ConversationSummary
from app.services.conversation_store_service import ConversationStoreService


def test_conversation_store_persists_messages(tmp_path):
    db_path = tmp_path / "conversation.db"
    import app.config as config_module

    original_enabled = config_module.config.conversation_persistence_enabled
    original_db_path = config_module.config.conversation_db_path
    config_module.config.conversation_persistence_enabled = True
    config_module.config.conversation_db_path = str(db_path)
    try:
        store = ConversationStoreService()
        store.append_message("session-1", "user", "你好")
        store.append_message("session-1", "assistant", "你好，有什么可以帮你？")

        history = store.get_messages("session-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[1].message_index == 2
    finally:
        config_module.config.conversation_persistence_enabled = original_enabled
        config_module.config.conversation_db_path = original_db_path


def test_conversation_store_summary_roundtrip(tmp_path):
    db_path = tmp_path / "summary.db"
    import app.config as config_module

    original_enabled = config_module.config.conversation_persistence_enabled
    original_db_path = config_module.config.conversation_db_path
    config_module.config.conversation_persistence_enabled = True
    config_module.config.conversation_db_path = str(db_path)
    try:
        store = ConversationStoreService()
        store.upsert_summary("session-2", "这是摘要", 6)

        summary = store.get_summary("session-2")
        assert isinstance(summary, ConversationSummary)
        assert summary.summary == "这是摘要"
        assert summary.last_message_index == 6
    finally:
        config_module.config.conversation_persistence_enabled = original_enabled
        config_module.config.conversation_db_path = original_db_path

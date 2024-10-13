from django.apps import AppConfig


class AnalyzeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "analyze"

    def ready(self):
        import analyze.signals  # Ensure this is correct
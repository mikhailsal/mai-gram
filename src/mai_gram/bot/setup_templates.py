"""Template selection and params UI for the setup workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.bot.setup_workflow import SetupSession
    from mai_gram.config import BotConfig, Settings
    from mai_gram.messenger.base import Messenger
    from mai_gram.response_templates.base import ResponseTemplate, TemplateGroup


def get_available_templates_for_bot(settings: Settings, bot_config: BotConfig | None) -> list[str]:
    all_templates = settings.get_available_templates()
    if bot_config and bot_config.allowed_templates:
        bot_set = set(bot_config.allowed_templates)
        return [t for t in all_templates if t in bot_set]
    return all_templates


async def show_template_group_selection(
    session: SetupSession,
    messenger: Messenger,
    settings: Settings,
    bot_config: BotConfig | None,
) -> bool:
    """Show the template group selection or return False if trivial."""
    templates = get_available_templates_for_bot(settings, bot_config)

    if len(templates) <= 1:
        session.selected_template = None
        return False

    from mai_gram.response_templates.registry import (
        get_template,
        get_templates_in_group,
        list_groups,
    )

    available_set = set(templates)
    groups_with_templates = []
    for grp in list_groups():
        grp_templates = [t for t in get_templates_in_group(grp.id) if t.name in available_set]
        if grp_templates:
            groups_with_templates.append((grp, grp_templates))

    ungrouped = [get_template(name) for name in templates if get_template(name).group == ""]

    total_choices = len(groups_with_templates) + len(ungrouped)
    if total_choices <= 1 and not groups_with_templates:
        session.selected_template = None
        return False

    from mai_gram.bot.setup_workflow import SetupState

    session.state = SetupState.CHOOSING_TEMPLATE_GROUP
    keyboard_rows = _build_group_keyboard(ungrouped, groups_with_templates)

    prompt_preview = session.selected_prompt_text[:80]
    if len(session.selected_prompt_text) > 80:
        prompt_preview += "..."

    await messenger.send_message(
        OutgoingMessage(
            text=(
                f"Model: {session.selected_model}\n"
                f"Prompt: {prompt_preview}\n\n"
                "Choose a response format category:"
            ),
            chat_id=session.chat_id,
            keyboard=messenger.build_inline_keyboard(keyboard_rows),
        )
    )
    return True


def _build_group_keyboard(
    ungrouped: list[ResponseTemplate],
    groups_with_templates: list[tuple[TemplateGroup, list[ResponseTemplate]]],
) -> list[list[tuple[str, str]]]:
    keyboard_rows: list[list[tuple[str, str]]] = []
    for tpl in ungrouped:
        label = f"{tpl.description} [default]" if tpl.name == "empty" else tpl.description
        keyboard_rows.append([(label, f"tpl_group:__single__:{tpl.name}")])
    for grp, grp_templates in groups_with_templates:
        count = len(grp_templates)
        label = f"{grp.label} ({count} variant{'s' if count != 1 else ''})"
        keyboard_rows.append([(label, f"tpl_group:{grp.id}")])
    return keyboard_rows


async def show_template_params_summary(
    session: SetupSession,
    tpl: ResponseTemplate,
    messenger: Messenger,
) -> None:
    """Show current template param defaults and let the user accept or configure."""
    from mai_gram.bot.setup_workflow import SetupState

    session.state = SetupState.CONFIGURING_TEMPLATE_PARAMS
    params = tpl.get_params()
    lines = [f"Template: {tpl.description}\n\nConfigurable parameters:"]
    for p in params:
        hint = ""
        if p.suggestions:
            hint = f"\n  options: {', '.join(p.suggestions)}"
        elif p.param_type == "int" and p.min_value is not None and p.max_value is not None:
            hint = f"\n  range: {p.min_value}-{p.max_value}"
        lines.append(f"• {p.key} = {p.default}  ({p.label}){hint}")

    lines.append("\nTo customize, type key=value pairs, one per line:")
    example_lines = "\n".join(f"{p.key}={p.default}" for p in params)
    lines.append(example_lines)

    keyboard_rows = [[("Use defaults", "tpl_params:__defaults__")]]
    await messenger.send_message(
        OutgoingMessage(
            text="\n".join(lines),
            chat_id=session.chat_id,
            keyboard=messenger.build_inline_keyboard(keyboard_rows),
        )
    )

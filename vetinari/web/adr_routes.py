"""ADR (Architecture Decision Records) API routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vetinari.web import is_admin_user, validate_json_fields

bp = Blueprint("adr", __name__)


@bp.route("/api/adr", methods=["GET"])
def api_adr_list():
    """Api adr list.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        status = request.args.get("status")
        category = request.args.get("category")
        limit = int(request.args.get("limit", 50))

        adrs = adr_system.list_adrs(status=status, category=category, limit=limit)

        return jsonify({"adrs": [a.to_dict() for a in adrs], "total": len(adrs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/<adr_id>", methods=["GET"])
def api_adr_get(adr_id):
    """Api adr get.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        adr = adr_system.get_adr(adr_id)

        if not adr:
            return jsonify({"error": "ADR not found"}), 404

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr", methods=["POST"])
def api_adr_create():
    """Api adr create.

    Returns:
        Tuple of results.
    """
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.adr import adr_system

        data = request.json or {}

        ok, err = validate_json_fields(data, ["title"])
        if not ok:
            return err

        adr = adr_system.create_adr(
            title=data.get("title"),
            category=data.get("category", "architecture"),
            context=data.get("context", ""),
            decision=data.get("decision", ""),
            consequences=data.get("consequences", ""),
            created_by=data.get("created_by", "user"),
        )

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/<adr_id>", methods=["PUT"])
def api_adr_update(adr_id):
    """Api adr update.

    Returns:
        Tuple of results.
    """
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.adr import adr_system

        data = request.json or {}

        adr = adr_system.update_adr(adr_id, data)

        if not adr:
            return jsonify({"error": "ADR not found"}), 404

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/<adr_id>/deprecate", methods=["POST"])
def api_adr_deprecate(adr_id):
    """Api adr deprecate.

    Returns:
        Tuple of results.
    """
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.adr import adr_system

        data = request.json or {}

        replacement_id = data.get("replacement_id")

        adr = adr_system.deprecate_adr(adr_id, replacement_id)

        if not adr:
            return jsonify({"error": "ADR not found"}), 404

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/propose", methods=["POST"])
def api_adr_propose():
    """Api adr propose.

    Returns:
        Tuple of results.
    """
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.adr import adr_system

        data = request.json or {}

        context = data.get("context", "")
        num_options = int(data.get("num_options", 3))

        proposal = adr_system.generate_proposal(context, num_options)

        return jsonify(
            {
                "question": proposal.question,
                "options": proposal.options,
                "recommended": proposal.recommended,
                "rationale": proposal.rationale,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/propose/accept", methods=["POST"])
def api_adr_propose_accept():
    """Api adr propose accept.

    Returns:
        Tuple of results.
    """
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.adr import adr_system

        data = request.json or {}

        question = data.get("question", "")
        options = data.get("options", [])
        recommended = data.get("recommended", 0)
        title = data.get("title", "Proposed Decision")
        category = data.get("category", "architecture")

        from vetinari.adr import ADRProposal

        proposal = ADRProposal(question=question, options=options, recommended=recommended)

        adr = adr_system.accept_proposal(proposal, title, category)

        return jsonify(adr.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/statistics", methods=["GET"])
def api_adr_statistics():
    """Api adr statistics.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        stats = adr_system.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/adr/is-high-stakes", methods=["GET"])
def api_adr_is_high_stakes():
    """Api adr is high stakes.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.adr import adr_system

        category = request.args.get("category", "architecture")
        is_high_stakes = adr_system.is_high_stakes(category)
        return jsonify({"is_high_stakes": is_high_stakes, "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""Admin and credentials API routes."""

from flask import Blueprint, jsonify, request
from vetinari.web import is_admin_user

bp = Blueprint('admin', __name__)


@bp.route('/api/admin/credentials', methods=['GET'])
def api_admin_list_credentials():
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        credentials = credential_manager.list()
        health = credential_manager.health()

        return jsonify({
            "status": "ok",
            "credentials": credentials,
            "health": health
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/admin/credentials/<source_type>', methods=['POST'])
def api_admin_set_credential(source_type):
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        data = request.json or {}
        token = data.get('token', '')
        credential_type = data.get('credential_type', 'bearer')
        rotation_days = data.get('rotation_days', 30)
        note = data.get('note', '')

        if not token:
            return jsonify({"error": "Token is required"}), 400

        credential_manager.set_credential(
            source_type=source_type,
            token=token,
            credential_type=credential_type,
            rotation_days=rotation_days,
            note=note
        )

        return jsonify({
            "status": "ok",
            "message": f"Credential set for {source_type}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/admin/credentials/<source_type>/rotate', methods=['POST'])
def api_admin_rotate_credential(source_type):
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        data = request.json or {}
        new_token = data.get('token', '')

        if not new_token:
            return jsonify({"error": "New token is required"}), 400

        success = credential_manager.rotate(source_type, new_token)

        if success:
            return jsonify({
                "status": "ok",
                "message": f"Credential rotated for {source_type}"
            })
        else:
            return jsonify({"error": "Credential not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/admin/credentials/<source_type>', methods=['DELETE'])
def api_admin_delete_credential(source_type):
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        credential_manager.vault.remove_credential(source_type)

        return jsonify({
            "status": "ok",
            "message": f"Credential removed for {source_type}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/admin/credentials/health', methods=['GET'])
def api_admin_credentials_health():
    if not is_admin_user():
        return jsonify({"error": "Admin privileges required"}), 403
    try:
        from vetinari.credentials import credential_manager

        health = credential_manager.health()

        return jsonify({
            "status": "ok",
            "health": health
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/admin/permissions', methods=['GET'])
def api_admin_permissions():
    return jsonify({"admin": is_admin_user()}), 200

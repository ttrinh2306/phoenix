import atexit
import codecs
import logging
import os
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import List, Optional
from urllib.parse import urljoin

from jinja2 import BaseLoader, Environment
from uvicorn import Config, Server

import phoenix.trace.v1 as pb
from phoenix.config import (
    EXPORT_DIR,
    get_env_access_token_expiry,
    get_env_auth_settings,
    get_env_database_connection_str,
    get_env_database_schema,
    get_env_enable_prometheus,
    get_env_grpc_port,
    get_env_host,
    get_env_host_root_path,
    get_env_port,
    get_env_refresh_token_expiry,
    get_pids_path,
    get_working_dir,
)
from phoenix.core.model_schema_adapter import create_model_from_inferences
from phoenix.db import get_printable_db_url
from phoenix.inferences.fixtures import FIXTURES, get_inferences
from phoenix.inferences.inferences import EMPTY_INFERENCES, Inferences
from phoenix.pointcloud.umap_parameters import (
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_N_SAMPLES,
    UMAPParameters,
)
from phoenix.server.app import (
    ScaffolderConfig,
    _db,
    create_app,
    create_engine_and_run_migrations,
    instrument_engine_if_enabled,
)
from phoenix.server.types import DbSessionFactory
from phoenix.settings import Settings
from phoenix.trace.fixtures import (
    TRACES_FIXTURES,
    get_dataset_fixtures,
    get_evals_from_fixture,
    get_trace_fixtures_by_project_name,
    load_example_traces,
    reset_fixture_span_ids_and_timestamps,
    send_dataset_fixtures,
)
from phoenix.trace.otel import decode_otlp_span, encode_span_to_otlp
from phoenix.trace.schemas import Span

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_WELCOME_MESSAGE = Environment(loader=BaseLoader()).from_string("""

██████╗ ██╗  ██╗ ██████╗ ███████╗███╗   ██╗██╗██╗  ██╗
██╔══██╗██║  ██║██╔═══██╗██╔════╝████╗  ██║██║╚██╗██╔╝
██████╔╝███████║██║   ██║█████╗  ██╔██╗ ██║██║ ╚███╔╝
██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║██║ ██╔██╗
██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║██║██╔╝ ██╗
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝ v{{ version }}

|
|  🌎 Join our Community 🌎
|  https://join.slack.com/t/arize-ai/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q
|
|  ⭐️ Leave us a Star ⭐️
|  https://github.com/Arize-ai/phoenix
|
|  📚 Documentation 📚
|  https://docs.arize.com/phoenix
|
|  🚀 Phoenix Server 🚀
|  Phoenix UI: {{ ui_path }}
|  Log traces:
|    - gRPC: {{ grpc_path }}
|    - HTTP: {{ http_path }}
|  Storage: {{ storage }}
{% if schema -%}
|    - Schema: {{ schema }}
{% endif -%}
""")

_EXPERIMENTAL_WARNING = """
🚨 WARNING: Phoenix is running in experimental mode. 🚨
|  Authentication enabled: {auth_enabled}
"""


def _write_pid_file_when_ready(
    server: Server,
    wait_up_to_seconds: float = 5,
) -> None:
    """Write PID file after server is started (or when time is up)."""
    time_limit = time() + wait_up_to_seconds
    while time() < time_limit and not server.should_exit and not server.started:
        sleep(1e-3)
    if time() >= time_limit and not server.started:
        server.should_exit = True
    _get_pid_file().touch()


def _remove_pid_file() -> None:
    _get_pid_file().unlink(missing_ok=True)


def _get_pid_file() -> Path:
    return get_pids_path() / str(os.getpid())


DEFAULT_UMAP_PARAMS_STR = f"{DEFAULT_MIN_DIST},{DEFAULT_N_NEIGHBORS},{DEFAULT_N_SAMPLES}"

if __name__ == "__main__":
    primary_inferences_name: str
    reference_inferences_name: Optional[str]
    trace_dataset_name: Optional[str] = None
    simulate_streaming: Optional[bool] = None

    primary_inferences: Inferences = EMPTY_INFERENCES
    reference_inferences: Optional[Inferences] = None
    corpus_inferences: Optional[Inferences] = None

    # Initialize the settings for the Server
    Settings.log_migrations = True

    # automatically remove the pid file when the process is being gracefully terminated
    atexit.register(_remove_pid_file)

    parser = ArgumentParser()
    parser.add_argument("--database-url", required=False)
    parser.add_argument("--export_path")
    parser.add_argument("--host", type=str, required=False)
    parser.add_argument("--port", type=int, required=False)
    parser.add_argument("--read-only", action="store_true", required=False)  # Default is False
    parser.add_argument("--no-internet", action="store_true")
    parser.add_argument("--umap_params", type=str, required=False, default=DEFAULT_UMAP_PARAMS_STR)
    parser.add_argument("--debug", action="store_true")
    # Whether the app is running in a development environment
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--no-ui", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument(
        "--with-fixture",
        type=str,
        required=False,
        default="",
        help=("Name of an inference fixture. Example: 'fixture1'"),
    )
    serve_parser.add_argument(
        "--with-trace-fixtures",
        type=str,
        required=False,
        default="",
        help=(
            "Comma separated list of tracing fixture names (spaces are ignored). "
            "Example: 'fixture1, fixture2'"
        ),
    )
    serve_parser.add_argument(
        "--with-projects",
        type=str,
        required=False,
        default="",
        help=(
            "Comma separated list of project names (spaces are ignored). "
            "Example: 'project1, project2'"
        ),
    )
    serve_parser.add_argument(
        "--force-fixture-ingestion",
        action="store_true",  # default is False
        required=False,
        help=(
            "Whether or not to check the database age before adding the fixtures. "
            "Default is False, i.e., fixtures will only be added if the "
            "database is new."
        ),
    )
    serve_parser.add_argument(
        "--scaffold-datasets",
        action="store_true",  # default is False
        required=False,
        help=(
            "Whether or not to add any datasets defined in "
            "the inputted project or trace fixture. "
            "Default is False. "
        ),
    )

    datasets_parser = subparsers.add_parser("datasets")
    datasets_parser.add_argument("--primary", type=str, required=True)
    datasets_parser.add_argument("--reference", type=str, required=False)
    datasets_parser.add_argument("--corpus", type=str, required=False)
    datasets_parser.add_argument("--trace", type=str, required=False)

    fixture_parser = subparsers.add_parser("fixture")
    fixture_parser.add_argument("fixture", type=str, choices=[fixture.name for fixture in FIXTURES])
    fixture_parser.add_argument("--primary-only", action="store_true")  # Default is False

    trace_fixture_parser = subparsers.add_parser("trace-fixture")
    trace_fixture_parser.add_argument(
        "fixture", type=str, choices=[fixture.name for fixture in TRACES_FIXTURES]
    )
    trace_fixture_parser.add_argument(
        "--simulate-streaming", action="store_true"
    )  # Default is False

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("fixture", type=str, choices=[fixture.name for fixture in FIXTURES])
    demo_parser.add_argument(
        "trace_fixture", type=str, choices=[fixture.name for fixture in TRACES_FIXTURES]
    )
    demo_parser.add_argument("--simulate-streaming", action="store_true")

    args = parser.parse_args()
    db_connection_str = (
        args.database_url if args.database_url else get_env_database_connection_str()
    )
    export_path = Path(args.export_path) if args.export_path else EXPORT_DIR

    force_fixture_ingestion = False
    scaffold_datasets = False
    tracing_fixture_names = set()
    if args.command == "datasets":
        primary_inferences_name = args.primary
        reference_inferences_name = args.reference
        corpus_inferences_name = args.corpus
        primary_inferences = Inferences.from_name(primary_inferences_name)
        reference_inferences = (
            Inferences.from_name(reference_inferences_name)
            if reference_inferences_name is not None
            else None
        )
        corpus_inferences = (
            None if corpus_inferences_name is None else Inferences.from_name(corpus_inferences_name)
        )
    elif args.command == "fixture":
        fixture_name = args.fixture
        primary_only = args.primary_only
        primary_inferences, reference_inferences, corpus_inferences = get_inferences(
            fixture_name,
            args.no_internet,
        )
        if primary_only:
            reference_inferences_name = None
            reference_inferences = None
    elif args.command == "trace-fixture":
        trace_dataset_name = args.fixture
        simulate_streaming = args.simulate_streaming
    elif args.command == "demo":
        fixture_name = args.fixture
        primary_inferences, reference_inferences, corpus_inferences = get_inferences(
            fixture_name,
            args.no_internet,
        )
        trace_dataset_name = args.trace_fixture
        simulate_streaming = args.simulate_streaming
    elif args.command == "serve":
        # We use sets to avoid duplicates
        if args.with_fixture:
            primary_inferences, reference_inferences, corpus_inferences = get_inferences(
                str(args.with_fixture),
                args.no_internet,
            )
        if args.with_trace_fixtures:
            tracing_fixture_names.update(
                [name.strip() for name in args.with_trace_fixtures.split(",")]
            )
        if args.with_projects:
            project_names = [name.strip() for name in args.with_projects.split(",")]
            tracing_fixture_names.update(
                fixture.name
                for name in project_names
                for fixture in get_trace_fixtures_by_project_name(name)
            )
        force_fixture_ingestion = args.force_fixture_ingestion
        scaffold_datasets = args.scaffold_datasets
    host: Optional[str] = args.host or get_env_host()
    display_host = host or "localhost"
    # If the host is "::", the convention is to bind to all interfaces. However, uvicorn
    # does not support this directly unless the host is set to None.
    if host and ":" in host:
        # format IPv6 hosts in brackets
        display_host = f"[{host}]"
    if host == "::":
        # TODO(dustin): why is this necessary? it's not type compliant
        host = None

    port = args.port or get_env_port()
    host_root_path = get_env_host_root_path()
    read_only = args.read_only

    model = create_model_from_inferences(
        primary_inferences,
        reference_inferences,
    )

    authentication_enabled, secret = get_env_auth_settings()

    fixture_spans: List[Span] = []
    fixture_evals: List[pb.Evaluation] = []
    if trace_dataset_name is not None:
        fixture_spans, fixture_evals = reset_fixture_span_ids_and_timestamps(
            (
                # Apply `encode` here because legacy jsonl files contains UUIDs as strings.
                # `encode` removes the hyphens in the UUIDs.
                decode_otlp_span(encode_span_to_otlp(span))
                for span in load_example_traces(trace_dataset_name).to_spans()
            ),
            get_evals_from_fixture(trace_dataset_name),
        )
        dataset_fixtures = list(get_dataset_fixtures(trace_dataset_name))
        if not read_only:
            Thread(
                target=send_dataset_fixtures,
                args=(f"http://{host}:{port}", dataset_fixtures),
            ).start()
    umap_params_list = args.umap_params.split(",")
    umap_params = UMAPParameters(
        min_dist=float(umap_params_list[0]),
        n_neighbors=int(umap_params_list[1]),
        n_samples=int(umap_params_list[2]),
    )

    logger.info(f"Server umap params: {umap_params}")
    if enable_prometheus := get_env_enable_prometheus():
        from phoenix.server.prometheus import start_prometheus

        start_prometheus()

    working_dir = get_working_dir().resolve()
    engine = create_engine_and_run_migrations(db_connection_str)
    instrumentation_cleanups = instrument_engine_if_enabled(engine)
    factory = DbSessionFactory(db=_db(engine), dialect=engine.dialect.name)
    corpus_model = (
        None if corpus_inferences is None else create_model_from_inferences(corpus_inferences)
    )
    # Print information about the server
    root_path = urljoin(f"http://{host}:{port}", host_root_path)
    msg = _WELCOME_MESSAGE.render(
        version=version("arize-phoenix"),
        ui_path=root_path,
        grpc_path=f"http://{host}:{get_env_grpc_port()}",
        http_path=urljoin(root_path, "v1/traces"),
        storage=get_printable_db_url(db_connection_str),
        schema=get_env_database_schema(),
    )
    if authentication_enabled:
        msg += _EXPERIMENTAL_WARNING.format(auth_enabled=True)
    if sys.platform.startswith("win"):
        msg = codecs.encode(msg, "ascii", errors="ignore").decode("ascii").strip()
    scaffolder_config = ScaffolderConfig(
        db=factory,
        tracing_fixture_names=tracing_fixture_names,
        force_fixture_ingestion=force_fixture_ingestion,
        scaffold_datasets=scaffold_datasets,
        phoenix_url=root_path,
    )
    app = create_app(
        db=factory,
        export_path=export_path,
        model=model,
        authentication_enabled=authentication_enabled,
        umap_params=umap_params,
        corpus=corpus_model,
        debug=args.debug,
        dev=args.dev,
        serve_ui=not args.no_ui,
        read_only=read_only,
        enable_prometheus=enable_prometheus,
        initial_spans=fixture_spans,
        initial_evaluations=fixture_evals,
        startup_callbacks=[lambda: print(msg)],
        shutdown_callbacks=instrumentation_cleanups,
        secret=secret,
        access_token_expiry=get_env_access_token_expiry(),
        refresh_token_expiry=get_env_refresh_token_expiry(),
        scaffolder_config=scaffolder_config,
    )
    server = Server(config=Config(app, host=host, port=port, root_path=host_root_path))  # type: ignore
    Thread(target=_write_pid_file_when_ready, args=(server,), daemon=True).start()

    # Start the server
    server.run()

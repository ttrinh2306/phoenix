from datetime import datetime

import pytest
import pytz
from phoenix.db import models
from sqlalchemy import insert
from strawberry.relay import GlobalID


async def test_compare_experiments_returns_expected_comparisons(
    test_client, comparison_experiments
):
    query = """
      query ($experimentIds: [GlobalID!]!) {
        compareExperiments(
          experimentIds: $experimentIds
        ) {
          example {
            id
            revision {
              input
              output
              metadata
            }
          }
          runComparisonItems {
            experimentId
            runs {
              id
              output
            }
          }
        }
      }
    """
    response = await test_client.post(
        "/graphql",
        json={
            "query": query,
            "variables": {
                "experimentIds": [
                    str(GlobalID("Experiment", str(2))),
                    str(GlobalID("Experiment", str(1))),
                    str(GlobalID("Experiment", str(3))),
                ],
            },
        },
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json.get("errors") is None
    assert response_json["data"] == {
        "compareExperiments": [
            {
                "example": {
                    "id": str(GlobalID("DatasetExample", str(2))),
                    "revision": {
                        "input": {"revision-4-input-key": "revision-4-input-value"},
                        "output": {"revision-4-output-key": "revision-4-output-value"},
                        "metadata": {"revision-4-metadata-key": "revision-4-metadata-value"},
                    },
                },
                "runComparisonItems": [
                    {
                        "experimentId": str(GlobalID("Experiment", str(2))),
                        "runs": [
                            {
                                "id": str(GlobalID("ExperimentRun", str(4))),
                                "output": "",
                            },
                        ],
                    },
                    {
                        "experimentId": str(GlobalID("Experiment", str(1))),
                        "runs": [],
                    },
                    {
                        "experimentId": str(GlobalID("Experiment", str(3))),
                        "runs": [
                            {
                                "id": str(GlobalID("ExperimentRun", str(7))),
                                "output": "run-7-output-value",
                            },
                            {
                                "id": str(GlobalID("ExperimentRun", str(8))),
                                "output": 8,
                            },
                        ],
                    },
                ],
            },
            {
                "example": {
                    "id": str(GlobalID("DatasetExample", str(1))),
                    "revision": {
                        "input": {"revision-2-input-key": "revision-2-input-value"},
                        "output": {"revision-2-output-key": "revision-2-output-value"},
                        "metadata": {"revision-2-metadata-key": "revision-2-metadata-value"},
                    },
                },
                "runComparisonItems": [
                    {
                        "experimentId": str(GlobalID("Experiment", str(2))),
                        "runs": [
                            {
                                "id": str(GlobalID("ExperimentRun", str(3))),
                                "output": 3,
                            },
                        ],
                    },
                    {
                        "experimentId": str(GlobalID("Experiment", str(1))),
                        "runs": [
                            {
                                "id": str(GlobalID("ExperimentRun", str(1))),
                                "output": {"output": "run-1-output-value"},
                            },
                        ],
                    },
                    {
                        "experimentId": str(GlobalID("Experiment", str(3))),
                        "runs": [
                            {
                                "id": str(GlobalID("ExperimentRun", str(5))),
                                "output": None,
                            },
                            {
                                "id": str(GlobalID("ExperimentRun", str(6))),
                                "output": {"output": "run-6-output-value"},
                            },
                        ],
                    },
                ],
            },
        ]
    }


@pytest.fixture
async def comparison_experiments(session):
    """
    Creates a dataset with four examples, three versions, and four experiments.

                Version 1   Version 2   Version 3
    Example 1   CREATED     PATCHED     PATCHED
    Example 2               CREATED
    Example 3   CREATED     DELETED
    Example 4                           CREATED

    Experiment 1: V1 (1 repetition)
    Experiment 2: V2 (1 repetition)
    Experiment 3: V2 (2 repetitions)
    Experiment 4: V3 (1 repetition)
    """

    dataset_id = await session.scalar(
        insert(models.Dataset)
        .returning(models.Dataset.id)
        .values(
            name="dataset-name",
            description="dataset-description",
            metadata_={"dataset-metadata-key": "dataset-metadata-value"},
        )
    )

    example_ids = (
        await session.scalars(
            insert(models.DatasetExample)
            .returning(models.DatasetExample.id)
            .values([{"dataset_id": dataset_id} for _ in range(4)])
        )
    ).all()

    version_ids = (
        await session.scalars(
            insert(models.DatasetVersion)
            .returning(models.DatasetVersion.id)
            .values(
                [
                    {
                        "dataset_id": dataset_id,
                        "description": f"version-{index}-description",
                        "metadata_": {
                            f"version-{index}-metadata-key": f"version-{index}-metadata-value"
                        },
                    }
                    for index in range(1, 4)
                ]
            )
        )
    ).all()

    await session.scalars(
        insert(models.DatasetExampleRevision)
        .returning(models.DatasetExampleRevision.id)
        .values(
            [
                {
                    **revision,
                    "input": {
                        f"revision-{revision_index + 1}-input-key": f"revision-{revision_index + 1}-input-value"  # noqa: E501
                    },
                    "output": {
                        f"revision-{revision_index + 1}-output-key": f"revision-{revision_index + 1}-output-value"  # noqa: E501
                    },
                    "metadata_": {
                        f"revision-{revision_index + 1}-metadata-key": f"revision-{revision_index + 1}-metadata-value"  # noqa: E501
                    },
                }
                for revision_index, revision in enumerate(
                    [
                        {
                            "dataset_example_id": example_ids[0],
                            "dataset_version_id": version_ids[0],
                            "revision_kind": "CREATE",
                        },
                        {
                            "dataset_example_id": example_ids[0],
                            "dataset_version_id": version_ids[1],
                            "revision_kind": "PATCH",
                        },
                        {
                            "dataset_example_id": example_ids[0],
                            "dataset_version_id": version_ids[2],
                            "revision_kind": "PATCH",
                        },
                        {
                            "dataset_example_id": example_ids[1],
                            "dataset_version_id": version_ids[1],
                            "revision_kind": "CREATE",
                        },
                        {
                            "dataset_example_id": example_ids[2],
                            "dataset_version_id": version_ids[0],
                            "revision_kind": "CREATE",
                        },
                        {
                            "dataset_example_id": example_ids[2],
                            "dataset_version_id": version_ids[1],
                            "revision_kind": "DELETE",
                        },
                        {
                            "dataset_example_id": example_ids[3],
                            "dataset_version_id": version_ids[2],
                            "revision_kind": "CREATE",
                        },
                    ]
                )
            ]
        )
    )

    experiment_ids = (
        await session.scalars(
            insert(models.Experiment)
            .returning(models.Experiment.id)
            .values(
                [
                    {
                        **experiment,
                        "name": f"experiment-{experiment_index + 1}-name",
                        "description": f"experiment-{experiment_index + 1}-description",
                        "repetitions": 1,
                        "metadata_": {
                            f"experiment-{experiment_index + 1}-metadata-key": f"experiment-{experiment_index + 1}-metadata-value"  # noqa: E501
                        },
                    }
                    for experiment_index, experiment in enumerate(
                        [
                            {
                                "dataset_id": dataset_id,
                                "dataset_version_id": version_ids[0],
                            },
                            {
                                "dataset_id": dataset_id,
                                "dataset_version_id": version_ids[1],
                            },
                            {
                                "dataset_id": dataset_id,
                                "dataset_version_id": version_ids[1],
                            },
                            {
                                "dataset_id": dataset_id,
                                "dataset_version_id": version_ids[0],
                            },
                        ]
                    )
                ]
            )
        )
    ).all()

    await session.scalars(
        insert(models.ExperimentRun)
        .returning(models.ExperimentRun.id)
        .values(
            [
                {
                    **run,
                    "output": [
                        {"result": {"output": f"run-{run_index + 1}-output-value"}},
                        {"result": f"run-{run_index + 1}-output-value"},
                        {"result": run_index + 1},
                        {"result": ""},
                        {},
                    ][run_index % 5],
                }
                for run_index, run in enumerate(
                    [
                        {
                            "experiment_id": experiment_ids[0],
                            "dataset_example_id": example_ids[0],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[0],
                            "dataset_example_id": example_ids[3],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[1],
                            "dataset_example_id": example_ids[0],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[1],
                            "dataset_example_id": example_ids[1],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[2],
                            "dataset_example_id": example_ids[0],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[2],
                            "dataset_example_id": example_ids[0],
                            "trace_id": None,
                            "repetition_number": 2,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[2],
                            "dataset_example_id": example_ids[1],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[2],
                            "dataset_example_id": example_ids[1],
                            "trace_id": None,
                            "repetition_number": 2,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[3],
                            "dataset_example_id": example_ids[0],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[3],
                            "dataset_example_id": example_ids[1],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                        {
                            "experiment_id": experiment_ids[3],
                            "dataset_example_id": example_ids[3],
                            "trace_id": None,
                            "repetition_number": 1,
                            "start_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "end_time": datetime(
                                year=2020, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc
                            ),
                            "error": None,
                        },
                    ]
                )
            ]
        )
    )

# based on https://github.com/BradyAJohnston/MolecularNodes/blob/80f3dac339750638bb5f5edc3b1e66292fccf642/molecularnodes/bpyd/attribute.py
# and this prompt: https://chatgpt.com/share/672de0c6-b18c-8013-84d1-ecf23043c790

import bpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Type, Any
from enum import Enum

# Retained Enums and Classes

class DomainType(Enum):
    POINT = "POINT"
    EDGE = "EDGE"
    FACE = "FACE"
    CORNER = "CORNER"
    CURVE = "CURVE"
    INSTANCE = "INSTANCE"
    LAYER = "LAYER"

class Domains:
    POINT = DomainType.POINT
    EDGE = DomainType.EDGE
    FACE = DomainType.FACE
    CORNER = DomainType.CORNER
    CURVE = DomainType.CURVE
    INSTANCE = DomainType.INSTANCE
    LAYER = DomainType.LAYER

@dataclass
class AttributeType:
    type_name: str
    value_name: str
    dtype: Type
    dimensions: tuple

    def __str__(self) -> str:
        return self.type_name

class AttributeTypes(Enum):
    FLOAT = AttributeType("FLOAT", "value", float, (1,))
    FLOAT_VECTOR = AttributeType("FLOAT_VECTOR", "vector", float, (3,))
    FLOAT2 = AttributeType("FLOAT2", "vector", float, (2,))
    FLOAT_COLOR = AttributeType("FLOAT_COLOR", "color", float, (4,))
    BYTE_COLOR = AttributeType("BYTE_COLOR", "color", int, (4,))
    QUATERNION = AttributeType("QUATERNION", "value", float, (4,))
    INT = AttributeType("INT", "value", int, (1,))
    INT8 = AttributeType("INT8", "value", int, (1,))
    INT32_2D = AttributeType("INT32_2D", "value", int, (2,))
    FLOAT4X4 = AttributeType("FLOAT4X4", "matrix", float, (4, 4))
    BOOLEAN = AttributeType("BOOLEAN", "value", bool, (1,))

def guess_attribute_type(series: pd.Series) -> AttributeType:
    """Guess the Blender attribute type based on the pandas series data type."""
    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        return AttributeTypes.INT.value
    elif pd.api.types.is_float_dtype(dtype):
        return AttributeTypes.FLOAT.value
    elif pd.api.types.is_bool_dtype(dtype):
        return AttributeTypes.BOOLEAN.value
    elif pd.api.types.is_object_dtype(dtype):
        # Check if the series contains list-like objects
        first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
        if isinstance(first_valid, (list, tuple, np.ndarray)):
            length = len(first_valid)
            if length == 2:
                return AttributeTypes.FLOAT2.value
            elif length == 3:
                return AttributeTypes.FLOAT_VECTOR.value
            elif length == 4:
                return AttributeTypes.FLOAT_COLOR.value
            elif length == 16:
                return AttributeTypes.FLOAT4X4.value
            else:
                return AttributeTypes.FLOAT_VECTOR.value
    return AttributeTypes.FLOAT.value

class PandasMesh:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        mesh_name: str = "PointCloudMeshwithAttributes",
        object_name: str = "PointCloudAttributes",
    ):
        self.mesh_name = mesh_name
        self.object_name = object_name
        self.included_columns = []  # To keep track of attributes added
        self.process_dataframe(dataframe)
        self.length = len(self.dataframe)
        self.vertices = [(0, 0, 0)] * self.length

        # Create the Blender mesh and object
        self.mesh = bpy.data.meshes.new(self.mesh_name)
        self.point_obj = bpy.data.objects.new(self.object_name, self.mesh)
        bpy.context.collection.objects.link(self.point_obj)
        self.mesh.from_pydata(self.vertices, [], [])
        self.mesh.update()

        # Add attributes from the dataframe
        self.add_attributes()

    def process_dataframe(self, dataframe: pd.DataFrame):
        """Process the dataframe to remove unsupported columns and prepare it for attribute creation."""
        included_columns = []
        excluded_columns = []

        for col in dataframe.columns:
            series = dataframe[col]
            dtype = series.dtype

            if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                included_columns.append(col)
            elif pd.api.types.is_object_dtype(dtype):
                first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
                if isinstance(first_valid, (list, tuple, np.ndarray)):
                    included_columns.append(col)
                else:
                    excluded_columns.append(col)
            else:
                excluded_columns.append(col)

        if excluded_columns:
            print(f"Columns not included (unsupported data types): {excluded_columns}")

        self.dataframe = dataframe[included_columns]
        self.included_columns = included_columns  # Store the included columns
        print(f"Columns added to the mesh: {included_columns}")

    def add_attributes(self):
        for column in self.dataframe.columns:
            series = self.dataframe[column]
            attr_type = guess_attribute_type(series)
            data = self.series_to_numpy(series, attr_type)
            self.store_attribute(self.point_obj, data, column, attr_type)

    def update(self, dataframe: pd.DataFrame):
        # Store previous included columns to know which attributes to remove
        previous_columns = set(self.included_columns)
        self.process_dataframe(dataframe)
        new_length = len(self.dataframe)
        if new_length != self.length:
            self.length = new_length
            self.vertices = [(0, 0, 0)] * self.length
            self.mesh.from_pydata(self.vertices, [], [])
            self.mesh.update()
        # Remove existing attributes that were previously added
        for attr_name in previous_columns:
            if attr_name in self.mesh.attributes:
                self.mesh.attributes.remove(self.mesh.attributes[attr_name])
        self.add_attributes()

    def clear(self):
        self.dataframe = pd.DataFrame()
        # Remove existing attributes that were added
        for attr_name in self.included_columns:
            if attr_name in self.mesh.attributes:
                self.mesh.attributes.remove(self.mesh.attributes[attr_name])
        self.included_columns = []

    def store_attribute(
        self,
        obj: bpy.types.Object,
        data: np.ndarray,
        name: str,
        attr_type: AttributeType,
        domain: DomainType = DomainType.POINT,
    ):
        attributes = obj.data.attributes
        if name in attributes:
            attribute = attributes[name]
        else:
            attribute = attributes.new(name, attr_type.type_name, domain.value)

        expected_length = len(attribute.data) * int(np.prod(attr_type.dimensions))
        if data.size != expected_length:
            raise ValueError(
                f"Data length {data.size} does not match expected size {expected_length} for attribute '{name}'."
            )

        attribute.data.foreach_set(attr_type.value_name, data.flatten())

    def series_to_numpy(self, series: pd.Series, attr_type: AttributeType) -> np.ndarray:
        if attr_type.dimensions == (1,):
            return series.to_numpy(dtype=attr_type.dtype)
        else:
            data = np.array(series.tolist(), dtype=attr_type.dtype)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=1)
            return data.reshape(-1, int(np.prod(attr_type.dimensions)))
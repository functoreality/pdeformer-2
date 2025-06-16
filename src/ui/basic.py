r"""User interface for PDEformer."""
from typing import Optional, Dict, Union, Any, List
from collections import deque
from numpy.typing import NDArray
from PyQt5.QtWidgets import QWidget
Value = Union[int, float, NDArray[float]]


class PDEDatabase:
    """
    A hierarchical database for PDE data, including types, coefficients, and additional arguments,
    where the structure can contain multiple nested sub-databases, similar to folders and files.
    """
    def __init__(self, name: str, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        """
        Initialize the hierarchical PDE database.
        """
        self.data_entries = {}
        self.sub_databases = {}
        self.name = name
        self.dependencies = {}
        self._validate_key(self.name)

    def get_value(self, key: str) -> Value:
        """
        Get the value of a data entry. The key can reference sub-databases using dot notation.

        Args:
            key (str): The key for the data entry, which can be in dot-separated format to traverse sub-databases.

        Returns:
            Value: The value of the data entry.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key not in self.data_entries:
                raise KeyError(f"Key '{key}' not found in data entries of '{self.name}'.")
            return self.data_entries[key]
        sub_db_name, rest_key = key_parts
        if sub_db_name not in self.sub_databases:
            raise KeyError(f"Sub-database '{sub_db_name}' not found in hierarchy of '{self.name}'.")
        return self.sub_databases[sub_db_name].get_value(rest_key)

    def set_value(self, key: str, value: Value) -> None:
        """
        Set the value of a data entry. The key can reference sub-databases using dot notation.

        Args:
            key (str): The key for the data entry, which can be in dot-separated format to traverse sub-databases.
            value (Value): The value to set for the data entry.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key in self.sub_databases:
                raise KeyError(f"Key '{key}' already exists as a sub-database in '{self.name}'.")
            self.data_entries[key] = value
        else:
            sub_db_name, rest_key = key_parts
            if sub_db_name not in self.sub_databases:
                sub_db = PDEDatabase(sub_db_name)
                self.sub_databases[sub_db_name] = sub_db
            self.sub_databases[sub_db_name].set_value(rest_key, value)

    def delete_value(self, key: str) -> None:
        """
        Delete a value entry from the database. The key can reference sub-databases using dot notation.

        Args:
            key (str): The key for the data entry to delete, which can be in
            dot-separated format to traverse sub-databases.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key in self.data_entries:
                del self.data_entries[key]
            else:
                raise KeyError(f"Key '{key}' not found in data entries of '{self.name}'.")
        else:
            sub_db_name, rest_key = key_parts
            if sub_db_name not in self.sub_databases:
                raise KeyError(f"Sub-database '{sub_db_name}' not found in hierarchy of '{self.name}'.")
            self.sub_databases[sub_db_name].delete_value(rest_key)

    def iter_auto_update(self, key: str) -> None:
        """
        Go through the hierarchy of the database and automatically update the value
        of a data entry based on its dependencies.

        Args:
            key (str): The key for the data entry to automatically update.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key in self.data_entries:
                self.auto_update(key)
            else:
                raise KeyError(f"Key '{key}' not found in data entries of '{self.name}'.")
        else:
            sub_db_name, rest_key = key_parts
            if sub_db_name not in self.sub_databases:
                raise KeyError(f"Sub-database '{sub_db_name}' not found in hierarchy of '{self.name}'.")
            self.auto_update(key)
            self.sub_databases[sub_db_name].iter_auto_update(rest_key)

    def auto_update(self, key: str) -> None:
        """
        Automatically update the value of a data entry based on its dependencies.

        Args:
            key (str): The key for the data entry to automatically update.
        """

    def get_subdatabase(self, key: str) -> 'PDEDatabase':
        """
        Get a sub-database by key. The key can reference nested sub-databases using dot notation.

        Args:
            key (str): The key for the sub-database, which can be in dot-separated format to traverse sub-databases.

        Returns:
            PDEDatabase: The requested sub-database.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key not in self.sub_databases:
                raise KeyError(f"Sub-database '{key}' not found in hierarchy of '{self.name}'.")
            return self.sub_databases[key]
        sub_db_name, rest_key = key_parts
        if sub_db_name not in self.sub_databases:
            raise KeyError(f"Sub-database '{sub_db_name}' not found in hierarchy of '{self.name}'.")
        return self.sub_databases[sub_db_name].get_subdatabase(rest_key)

    def set_subdatabase(self, key: str, sub_db: 'PDEDatabase') -> None:
        """
        Set a sub-database by key. The key can reference nested sub-databases using dot notation.

        Args:
            key (str): The key for the sub-database, which can be in dot-separated format to traverse sub-databases.
            sub_db (PDEDatabase): The sub-database to add.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key in self.data_entries:
                raise KeyError(f"Key '{key}' already exists as a data entry in '{self.name}'.")
            if sub_db.name != key:
                # Rename the sub-database to match the key
                sub_db.name = key
            self.sub_databases[key] = sub_db
            # add dependencies
            sub_dependencies = sub_db.dependencies
            for sub_key, sub_value_list in sub_dependencies.items():
                if not isinstance(sub_value_list, list):
                    raise ValueError(f"Dependencies must be a list of strings, but got '{sub_value_list}' instead.")
                if f"{key}.{sub_key}" not in self.dependencies:
                    self.dependencies[f"{key}.{sub_key}"] = []
                for sub_value in sub_value_list:
                    self.dependencies[f"{key}.{sub_key}"].append(f"{key}.{sub_value}")
        else:
            sub_db_name, rest_key = key_parts
            if sub_db_name not in self.sub_databases:
                sub_db = PDEDatabase(sub_db_name)
                self.sub_databases[sub_db_name] = sub_db
            self.sub_databases[sub_db_name].set_subdatabase(rest_key, sub_db)
            # add dependencies
            sub_dependencies = sub_db.dependencies
            for sub_key, sub_value_list in sub_dependencies.items():
                if not isinstance(sub_value_list, list):
                    raise ValueError(f"Dependencies must be a list of strings, but got '{sub_value_list}' instead.")
                if f"{key}.{sub_key}" not in self.dependencies:
                    self.dependencies[f"{key}.{sub_key}"] = []
                for sub_value in sub_value_list:
                    self.dependencies[f"{key}.{sub_key}"].append(f"{key}.{sub_value}")

    def delete_subdatabase(self, key: str) -> None:
        """
        Delete a sub-database by key. The key can reference nested sub-databases
        using dot notation.

        Args:
            key (str): The key for the sub-database to delete, which can be in
            dot-separated format to traverse sub-databases.
        """
        key_parts = key.split(".", 1)
        if len(key_parts) == 1:
            if key in self.sub_databases:
                del self.sub_databases[key]
            else:
                raise KeyError(f"Sub-database '{key}' not found in hierarchy.")
        else:
            sub_db_name, rest_key = key_parts
            if sub_db_name not in self.sub_databases:
                raise KeyError(f"Sub-database '{sub_db_name}' not found in hierarchy of '{self.name}'.")
            self.sub_databases[sub_db_name].delete_subdatabase(rest_key)

    def get_rep(self) -> Union[Dict, Value]:
        """
        Get a nested dictionary or value representation of the database.

        Returns:
            Union[Dict, Value]: Representation of the database.
        """
        nested_dict = {**self.data_entries}
        for sub_db_name, sub_db in self.sub_databases.items():
            nested_dict[sub_db_name] = sub_db.get_nested_dict()
        return nested_dict

    def _register_dependencies(self) -> Dict[str, List[str]]:
        """
        Register the dependencies for the database. This method should be explicitly
        called in subclasses after initialization.

        Returns:
            Dict[str, List[str]]: The dependencies for the database.
        """
        return {}

    def _validate_key(self, key: str) -> None:
        """
        Validate that the key does not contain invalid characters.

        Args:
            key (str): The key to validate.
        """
        if not key:
            raise ValueError("Key cannot be empty.")
        if '.' in key:
            raise ValueError("Key cannot contain the character '.' except for separating sub-databases.")

    def __repr__(self) -> str:
        return (f"PDEDatabase(name={self.name}, data_entries={self.data_entries}, "
                f"sub_databases={list(self.sub_databases.keys())})")


class UIElement(QWidget):
    r"""Basic class of UI elements.

    Args:
        name (str): The name of the UI element.
        ui_manager (UIManager): The manager of the UI elements.
        depends_on (Optional[Dict[str, str]]): Database keys that the UI element
            depends on. Each key-value pair is in the format of
            "identifier: database_key".
            Common identifiers include "visible" and "enabled", which are used to
            control the visibility and the enabled state of the UI element.
            Default: None.
        updates (Optional[Dict[str, str]]): Database keys that the UI element
            updates. Each key-value pair is in the format of
            "identifier: database_key". Default: None.
    """

    def __init__(self,  # pylint: disable=unused-argument
                 name: str,
                 ui_manager: "UIManager",
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__()
        self.name = name
        self.ui_manager = ui_manager
        self.depends_on = depends_on if depends_on else {}
        self.updates = updates if updates else {}
        self.ui_manager.register(self)

    def update(self) -> None:
        r"""Update the UI element according to the database. No additional
        arguments are passed."""
        if "visible" in self.depends_on:
            key = self.depends_on["visible"]
            visible = self.ui_manager.get_db_value(key)
            self.setVisible(visible)

    def update_database(self) -> None:
        r"""Update the database according to the UI element."""


class UIManager:
    r"""Dependency manager for UI elements and database."""

    def __init__(self, db: PDEDatabase) -> None:
        self.db = db
        self.ui_elements = {}
        self.data_updates = {}  # inverse relationship of depends_on
        self.toporder_updates = self._get_toporder_updates()

    def register(self,
                 ui_element: UIElement) -> None:
        r"""Register the UI element to the dependency manager."""
        if ui_element.name in self.ui_elements:
            raise ValueError(f"UI element with name '{ui_element.name}' already"
                             " exists in the dependency manager.")
        self.ui_elements[ui_element.name] = ui_element
        for dependency in ui_element.depends_on.values():
            if dependency not in self.data_updates:
                self.data_updates[dependency] = []
            if ui_element.name not in self.data_updates[dependency]:
                self.data_updates[dependency].append(ui_element.name)
        self.toporder_updates = self._get_toporder_updates()

    def update_ui(self, name: str, *args, **kwargs) -> None:
        r"""Update the UI element, and update the database and other UI elements
        accordingly."""
        # Update the specified UI element's database
        self.ui_elements[name].update_database(*args, **kwargs)
        sorted_entries = self.toporder_updates.get(name, [])

        # Update the database values
        for entry in sorted_entries:
            self.db.iter_auto_update(entry)

        # Update the UI elements
        all_data_entries = set(sorted_entries)
        all_data_entries.update(self.ui_elements[name].updates.values())
        all_ui_elements = set()
        for entry in all_data_entries:
            if entry in self.data_updates:
                all_ui_elements.update(self.data_updates[entry])

        for ui_name in all_ui_elements:
            self.ui_elements[ui_name].update()

    def _get_toporder_updates(self) -> Dict[str, List[str]]:
        r"""Get the topological order of the data entries that need to be updated when
        a UI element is updated."""
        toporder_updates = {}
        for ui_name, ui_element in self.ui_elements.items():
            # Identify all data entries that need to be updated
            all_data_entries = set(ui_element.updates.values())
            queue = deque(all_data_entries)
            while queue:
                current_entry = queue.popleft()
                # Retrieve all data entries that depend on the current_entry
                dependents = self.db.dependencies.get(current_entry, [])
                for dependent in dependents:
                    if dependent not in all_data_entries:
                        all_data_entries.add(dependent)
                        queue.append(dependent)
            # Topological sort
            dependency_graph = {entry: set() for entry in all_data_entries}
            for entry in all_data_entries:
                for key, dependents in self.db.dependencies.items():
                    if entry in dependents and key in all_data_entries:
                        dependency_graph[entry].add(key)
            no_dependency_queue = deque([entry for entry, deps in dependency_graph.items() if not deps])
            sorted_entries = []

            while no_dependency_queue:
                current = no_dependency_queue.popleft()
                sorted_entries.append(current)

                for dependent in self.db.dependencies.get(current, []):
                    dependency_graph[dependent].remove(current)
                    if not dependency_graph[dependent]:
                        no_dependency_queue.append(dependent)

            if len(sorted_entries) < len(all_data_entries):
                raise ValueError("Cycles detected in the dependency graph. Cannot perform topological sort.")
            toporder_updates[ui_name] = sorted_entries

        # Exclude the data entries that have been directly updated by the UI elements
        for ui_name, ui_element in self.ui_elements.items():
            for entry in ui_element.updates.values():
                if entry in toporder_updates[ui_name]:
                    toporder_updates[ui_name].remove(entry)

        return toporder_updates

    def update_all_dependencies(self) -> None:
        r"""Update all the database entries and UI elements."""
        # topological sort for all data entries
        all_data_entries = set(self.db.dependencies.keys())
        queue = deque(all_data_entries)
        while queue:
            current_entry = queue.popleft()
            # Retrieve all data entries that depend on the current_entry
            dependents = self.db.dependencies.get(current_entry, [])
            for dependent in dependents:
                if dependent not in all_data_entries:
                    all_data_entries.add(dependent)
                    queue.append(dependent)
        # Topological sort
        dependency_graph = {entry: set() for entry in all_data_entries}
        for entry in all_data_entries:
            for key, dependents in self.db.dependencies.items():
                if entry in dependents and key in all_data_entries:
                    dependency_graph[entry].add(key)
        no_dependency_queue = deque([entry for entry, deps in dependency_graph.items() if not deps])
        sorted_entries = []

        while no_dependency_queue:
            current = no_dependency_queue.popleft()
            sorted_entries.append(current)

            for dependent in self.db.dependencies.get(current, []):
                dependency_graph[dependent].remove(current)
                if not dependency_graph[dependent]:
                    no_dependency_queue.append(dependent)

        if len(sorted_entries) < len(all_data_entries):
            raise ValueError("Cycles detected in the dependency graph. Cannot perform topological sort.")

        # Update the database values
        for entry in sorted_entries:
            self.update_db_value(entry)

        # Update the UI elements
        for ui_name in self.ui_elements:
            self.ui_elements[ui_name].update()

    def update_db_value(self, key: str) -> None:
        r"""Auto-update the value of the given key in the database."""
        self.db.iter_auto_update(key)

    def update_db_subdatabase(self, key: str, sub_db: PDEDatabase) -> None:
        r"""Update the sub-database of the given key in the database, and update
        the UI elements that depend on this database key."""
        self.set_db_subdatabase(key, sub_db)
        if key in self.data_updates:
            for ui_name in self.data_updates[key]:
                self.ui_elements[ui_name].update()

    def set_db_value(self, key: str, value: Value) -> None:
        r"""Set the value of the given key in the database. Typically used by
        update_database methods of the UI elements. Do not update the UI elements
        that depend on this database key."""
        self.db.set_value(key, value)

    def get_db_value(self, key: str) -> Value:
        r"""Get the value of the given key in the database."""
        return self.db.get_value(key)

    def delete_db_value(self, key: str) -> None:
        r"""Delete the value of the given key in the database."""
        self.db.delete_value(key)

    def set_db_subdatabase(self, key: str, sub_db: PDEDatabase) -> None:
        r"""Set the sub-database of the given key in the database."""
        self.db.set_subdatabase(key, sub_db)

    def get_db_subdatabase(self, key: str) -> PDEDatabase:
        r"""Get the sub-database of the given key in the database."""
        return self.db.get_subdatabase(key)

    def delete_db_subdatabase(self, key: str) -> None:
        r"""Delete the sub-database of the given key in the database."""
        self.db.delete_subdatabase(key)

    def get_nested_dict_representation(self) -> Dict[str, Any]:
        r"""Get a nested dictionary representation of the entire database."""
        return self.db.get_nested_dict()

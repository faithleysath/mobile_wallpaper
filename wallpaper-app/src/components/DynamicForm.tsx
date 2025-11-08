import React, { useState, useEffect } from 'react';
import {
  IonItem,
  IonLabel,
  IonInput,
  IonToggle,
  IonRange,
  IonSelect,
  IonSelectOption,
  IonTextarea,
} from '@ionic/react';
import { ApiConfig } from '../services/APIService';

interface DynamicFormProps {
  apiConfig: ApiConfig | null;
  onFormChange: (payload: Record<string, string | number | boolean | string[]>) => void;
}

const DynamicForm: React.FC<DynamicFormProps> = ({ apiConfig, onFormChange }) => {
  const [formState, setFormState] = useState<Record<string, string | number | boolean | string[]>>({});

  useEffect(() => {
    if (apiConfig) {
      const initialState: Record<string, string | number | boolean | string[]> = {};
      let pathParamIndex = 0;
      apiConfig.parameters.forEach((param) => {
        if (param.enable !== false) {
          const key = param.name || `_path_${pathParamIndex++}`;
          initialState[key] = Array.isArray(param.value) ? param.value[0] : param.value;
        }
      });
      setFormState(initialState);
      onFormChange(initialState);
    }
  }, [apiConfig, onFormChange]);

  const handleInputChange = (name: string, value: string | number | boolean | string[]) => {
    const newState = { ...formState, [name]: value };
    setFormState(newState);
    onFormChange(newState);
  };

  if (!apiConfig) {
    return null;
  }

  return (
    <div>
      {apiConfig.parameters.map((param, index) => {
        const key = param.name || `_path_${index}`;
        if (param.enable === false) return null;

        switch (param.type) {
          case 'integer':
            return (
              <IonItem key={key}>
                <IonLabel>{param.friendly_name}</IonLabel>
                <IonRange
                  min={param.min_value}
                  max={param.max_value}
                  value={formState[key] as number}
                  onIonChange={(e) => handleInputChange(key, e.detail.value as number)}
                >
                  <IonInput
                    type="number"
                    value={formState[key] as number}
                    slot="end"
                    style={{ width: '60px', textAlign: 'right' }}
                    onIonChange={(e) => handleInputChange(key, parseInt(e.detail.value!, 10))}
                  />
                </IonRange>
              </IonItem>
            );
          case 'boolean':
            return (
              <IonItem key={key}>
                <IonLabel>{param.friendly_name}</IonLabel>
                <IonToggle
                  checked={!!formState[key]}
                  onIonChange={(e) => handleInputChange(key, e.detail.checked)}
                />
              </IonItem>
            );
          case 'enum':
            return (
              <IonItem key={key}>
                <IonLabel>{param.friendly_name}</IonLabel>
                <IonSelect
                  value={formState[key]}
                  onIonChange={(e) => handleInputChange(key, e.detail.value)}
                >
                  {(param.friendly_value || (Array.isArray(param.value) ? param.value : [])).map((val: string, i: number) => (
                    <IonSelectOption key={val} value={Array.isArray(param.value) ? param.value[i] : val}>
                      {val}
                    </IonSelectOption>
                  ))}
                </IonSelect>
              </IonItem>
            );
          case 'string':
          case 'list':
            return (
              <IonItem key={key}>
                <IonLabel position="stacked">{param.friendly_name}</IonLabel>
                <IonTextarea
                  value={String(formState[key] || (Array.isArray(param.value) ? param.value.join(param.split_str || '|') : param.value))}
                  onIonChange={(e) => handleInputChange(key, e.detail.value!)}
                />
              </IonItem>
            );
          default:
            return null;
        }
      })}
    </div>
  );
};

export default DynamicForm;

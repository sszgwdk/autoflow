'use client';

import { setDefault } from '@/api/commons';
import { deleteLlm, listLlms, type LLM } from '@/api/llms';
import { actions } from '@/components/cells/actions';
import { DataTableRemote } from '@/components/data-table-remote';
import { Badge } from '@/components/ui/badge';
import { getErrorMessage } from '@/lib/errors';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { TrashIcon } from 'lucide-react';
import Link from 'next/link';
import { toast } from 'sonner';

export function LLMsTable () {
  return (
    <DataTableRemote
      columns={columns}
      apiKey="api.llms.list"
      api={listLlms}
      idColumn="id"
    />
  );
}

const helper = createColumnHelper<LLM>();
const columns: ColumnDef<LLM, any>[] = [
  helper.accessor('id', {
    header: 'ID',
    cell: ({ row }) => row.original.id,
  }),
  helper.accessor('name', {
    header: 'NAME',
    cell: ({ row }) => {
      const { id, name, is_default } = row.original;
      return (
        <Link className="flex gap-1 items-center underline" href={`/llms/${id}`}>
          {is_default && <Badge>default</Badge>}
          {name}
        </Link>
      );
    },
  }),
  helper.display({
    header: 'PROVIDER / MODEL',
    cell: ({ row }) => {
      const { model, provider } = row.original;
      return (
        <>
          <strong>{provider}</strong>/<span>{model}</span>
        </>
      );
    },
  }),
  helper.display({
    id: 'Operations',
    header: 'ACTIONS',
    cell: actions(row => ([
      {
        key: 'set-default',
        title: 'Set Default',
        disabled: row.is_default,
        action: async (context) => {
          try {
            await setDefault('llms', row.id);
            context.table.reload?.();
            context.startTransition(() => {
              context.router.refresh();
            });
            context.setDropdownOpen(false);
            toast.success(`Successfully set default LLM to ${row.name}.`);
          } catch (e) {
            toast.error(`Failed to set default LLM to ${row.name}.`, {
              description: getErrorMessage(e),
            });
            throw e;
          }
        },
      },
      {
        key: 'delete',
        action: async ({ table, setDropdownOpen }) => {
          await deleteLlm(row.id);
          table.reload?.();
          setDropdownOpen(false);
        },
        title: 'Delete',
        icon: <TrashIcon className="size-3" />,
        dangerous: {},
      },
    ])),
  }),
];
